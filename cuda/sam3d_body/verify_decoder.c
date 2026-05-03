/*
 * verify_decoder (CUDA) — Step 5g: drive the full decoder forward loop
 * on GPU (decoder_layer + norm_final + heads + kp_token_update) with
 * the per-layer pose-output cycle (decode_pose_raw + MHR forward +
 * keypoints + camera_project) on CPU. Mirrors cpu/sam3d_body/
 * verify_decoder_full.c.
 *
 * Inputs:
 *   image_embeddings_after_ray.npy           (1, 1280, 32, 32) — encoder + ray_cond
 *   decoder_layer0_in__x.npy                 (1, 145, 1024) — initial tokens
 *   decoder_layer0_in__x_pe.npy              (1, 145, 1024) — initial augment
 *   decoder_layer0_in__context_pe.npy        (1, 1024, 1280) — image_pe in token form
 *   decoder_batch__{cam_int,bbox_*,*img_size,affine_trans}.npy — camera batch
 *
 * Diff targets (final / per-layer):
 *   head_pose_proj_raw.npy             (519,)
 *   head_camera_proj_raw.npy           (3,)
 *   decoder_pose_layer{0..4}__pred_keypoints_{2d_cropped,2d_depth,3d}.npy
 *   decoder_pose_layer{0..4}__pred_cam_t.npy
 *   decoder_pose_layer5__pred_keypoints_{2d_cropped,2d_depth,3d}.npy (final)
 *   decoder_pose_layer5__pred_cam_t.npy (final)
 */

#include "cuda_sam3d_body_runner.h"

/* CPU decoder + MHR implementations live in sam3d_body_cpu.c (linked
 * into the same binary). The runner.c is the unique provider of
 * safetensors impl symbols. Here we just need declarations. */
#include "../../common/sam3d_body_decoder.h"
#include "../../common/sam3d_body_mhr.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

static float *load_or_die(const char *refdir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8];
    float *d = (float *)npy_load(path, &nd, dims, NULL);
    if (!d) {
        fprintf(stderr, "[cuda verify_decoder] missing %s\n", path);
        return NULL;
    }
    return d;
}

static float *load_or_die_dims(const char *refdir, const char *name,
                               int *out_nd, int out_dims[8])
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    float *d = (float *)npy_load(path, out_nd, out_dims, NULL);
    if (!d) {
        fprintf(stderr, "[cuda verify_decoder] missing %s\n", path);
        return NULL;
    }
    return d;
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void resolve_variant_path(const char *dir, const char *bucket,
                                 cuda_sam3d_body_backbone_t backbone,
                                 char *out, size_t out_sz)
{
    const char *tag = (backbone == CUDA_SAM3D_BODY_BACKBONE_VITH) ?
        "vith" : "dinov3";
    snprintf(out, out_sz, "%s/sam3d_body_%s_%s.safetensors",
             dir, tag, bucket);
    if (file_exists(out)) return;
    snprintf(out, out_sz, "%s/sam3d_body_%s.safetensors", dir, bucket);
}

static int diff_pair(const char *label, const float *a, const float *b,
                     size_t n, float thresh)
{
    if (n == 0) {
        fprintf(stderr, "[cuda verify_decoder] %-44s empty diff FAIL\n",
                label);
        return 1;
    }
    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int fail = (mx >= thresh) || (mean >= thresh * 0.15f);
    fprintf(stderr, "[cuda verify_decoder] %-44s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *mhr_dir = NULL, *refdir = NULL;
    /* CPU floor at this stage: ~1e-4 max, ~1e-5 mean (FP32 LN+GEMM
     * accumulation chain spans 6 layers + heads). CUDA matches CPU
     * within 1e-5 in earlier per-stage verifies; budget set to match
     * verify_decoder_full's threshold so we catch real drift. */
    float threshold = 5e-3f;
    int device = 0, verbose = 0;
    int n_threads = 0;
    const char *precision = "bf16";
    cuda_sam3d_body_backbone_t backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = CUDA_SAM3D_BODY_BACKBONE_VITH;
            else {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "--device")          && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision")       && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --mhr-assets DIR "
                        "--refdir DIR [--threshold F] [--device N] "
                        "[--backbone dinov3|vith] [--precision bf16|fp16] "
                        "[-t N] [-v]\n", argv[0]);
        return 2;
    }

    /* CPU model (for MHR-in-the-loop helpers + init_pose/init_camera). */
    char p[1024], p2[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, p, sizeof(p));
    resolve_variant_path(sft_dir, "mhr_head", backbone, p2, sizeof(p2));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(p, p2);
    if (!m) { fprintf(stderr, "[cuda verify_decoder] decoder_load failed\n"); return 3; }

    snprintf(p,  sizeof(p),  "%s/sam3d_body_mhr_jit.safetensors", mhr_dir);
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_jit.json", mhr_dir);
    sam3d_body_mhr_assets *mhr = sam3d_body_mhr_load(p, p2);
    if (!mhr) {
        fprintf(stderr, "[cuda verify_decoder] mhr_load failed\n");
        sam3d_body_decoder_free(m); return 4;
    }

    /* CUDA ctx. */
    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = backbone,
    };
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[cuda verify_decoder] cuda_create failed\n");
        sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m); return 5;
    }

    const int Dc = m->kv_dim;          /* 1280 */
    const int D  = m->dim;             /* 1024 */
    const int K  = m->n_keypoints;     /* 70 */
    const int N_Q = 1 + 1 + 1 + m->n_hand_tokens + 2 * K; /* 145 */
    const int V  = 18439;
    const int J  = 127;
    const int NL = m->n_layers;        /* 6 */

    /* Reference inputs. */
    int img_nd = 0, img_dims[8] = {0};
    int pe_nd = 0, pe_dims[8] = {0};
    float *img_emb     = load_or_die_dims(refdir, "image_embeddings_after_ray",
                                          &img_nd, img_dims);
    float *init_x      = load_or_die(refdir, "decoder_layer0_in__x");
    float *init_xpe    = load_or_die(refdir, "decoder_layer0_in__x_pe");
    float *ctx_pe_tok  = load_or_die_dims(refdir, "decoder_layer0_in__context_pe",
                                          &pe_nd, pe_dims);
    if (!img_emb || !init_x || !init_xpe || !ctx_pe_tok) {
        free(img_emb); free(init_x); free(init_xpe); free(ctx_pe_tok);
        cuda_sam3d_body_destroy(ctx);
        sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m); return 6;
    }
    if (img_nd != 4 || pe_nd != 3 || img_dims[0] != 1 ||
        img_dims[1] != Dc || pe_dims[0] != 1 ||
        pe_dims[1] != img_dims[2] * img_dims[3] ||
        pe_dims[2] != Dc) {
        fprintf(stderr,
                "[cuda verify_decoder] bad image/context shape: "
                "image rank=%d dims=(%d,%d,%d,%d), context_pe rank=%d "
                "dims=(%d,%d,%d), Dc=%d\n",
                img_nd, img_dims[0], img_dims[1], img_dims[2], img_dims[3],
                pe_nd, pe_dims[0], pe_dims[1], pe_dims[2], Dc);
        free(img_emb); free(init_x); free(init_xpe); free(ctx_pe_tok);
        cuda_sam3d_body_destroy(ctx);
        sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m); return 6;
    }
    const int H = img_dims[2];
    const int W = img_dims[3];
    const int N_C = H * W;
    /* Flatten image_emb (CHW) → ctx_in (HW × C) for decoder_layer context_in. */
    float *ctx_in = (float *)malloc((size_t)N_C * Dc * sizeof(float));
    /* ctx_pe is dumped in token form (HW, C); permute to CHW for kp_token_update
     * is not needed since kp_token_update takes image_emb_chw (which is img_emb). */
    if (!ctx_in) {
        free(img_emb); free(init_x); free(init_xpe); free(ctx_pe_tok);
        cuda_sam3d_body_destroy(ctx);
        sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m); return 6;
    }
    for (int n = 0; n < N_C; n++)
        for (int c = 0; c < Dc; c++)
            ctx_in[(size_t)n * Dc + c] = img_emb[(size_t)c * N_C + n];

    /* Camera batch from ref. */
    float *cam_int       = load_or_die(refdir, "decoder_batch__cam_int");
    float *bbox_center   = load_or_die(refdir, "decoder_batch__bbox_center");
    float *bbox_scale    = load_or_die(refdir, "decoder_batch__bbox_scale");
    float *ori_img_size  = load_or_die(refdir, "decoder_batch__ori_img_size");
    float *img_size      = load_or_die(refdir, "decoder_batch__img_size");
    float *affine_trans  = load_or_die(refdir, "decoder_batch__affine_trans");
    if (!cam_int || !bbox_center || !bbox_scale || !ori_img_size ||
        !img_size || !affine_trans) {
        free(img_emb); free(init_x); free(init_xpe); free(ctx_pe_tok); free(ctx_in);
        free(cam_int); free(bbox_center); free(bbox_scale);
        free(ori_img_size); free(img_size); free(affine_trans);
        cuda_sam3d_body_destroy(ctx);
        sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m); return 6;
    }
    sam3d_body_camera_batch B;
    memcpy(B.cam_int,      cam_int,      9 * sizeof(float));
    memcpy(B.bbox_center,  bbox_center,  2 * sizeof(float));
    B.bbox_scale = bbox_scale[0];
    memcpy(B.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(B.img_size,     img_size,     2 * sizeof(float));
    memcpy(B.affine_trans, affine_trans, 6 * sizeof(float));
    B.use_intrin_center    = 0;
    B.default_scale_factor = 1.0f;
    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);

    /* Per-layer working buffers. */
    float *tokens   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *tokens_b = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *augment  = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *tokens_n = (float *)malloc((size_t)N_Q * D * sizeof(float));
    /* MHR scratch + outputs. */
    const size_t mhr_scratch_floats = (size_t)1 *
        (889 + (size_t)J*8*2 + (size_t)V*3*3);
    float *mhr_scratch = (float *)malloc(mhr_scratch_floats * sizeof(float));
    float *verts_cm = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *gskel_cm = (float *)malloc((size_t)J * 8 * sizeof(float));
    float *verts_m  = (float *)malloc((size_t)V * 3 * sizeof(float));
    float *jc_m     = (float *)malloc((size_t)J * 3 * sizeof(float));

    /* Per-layer debug snapshots for diff. */
    float *dbg_kp3d  = (float *)malloc((size_t)NL * K * 3 * sizeof(float));
    float *dbg_kp2dc = (float *)malloc((size_t)NL * K * 2 * sizeof(float));
    float *dbg_kp2dd = (float *)malloc((size_t)NL * K     * sizeof(float));
    float *dbg_camt  = (float *)malloc((size_t)NL * 3     * sizeof(float));

    if (!tokens || !tokens_b || !augment || !tokens_n ||
        !mhr_scratch || !verts_cm || !gskel_cm || !verts_m || !jc_m ||
        !dbg_kp3d || !dbg_kp2dc || !dbg_kp2dd || !dbg_camt) {
        fprintf(stderr, "[cuda verify_decoder] alloc failed\n");
        return 7;
    }

    memcpy(tokens,  init_x,   (size_t)N_Q * D * sizeof(float));
    memcpy(augment, init_xpe, (size_t)N_Q * D * sizeof(float));

    const float *ip = (const float *)m->init_pose.data;
    const float *ic = (const float *)m->init_camera.data;
    float pose_raw[519], cam_raw[3];
    float pose519[519], cam3[3];
    float mp_buf[204], shape_buf[45], face_buf[72];
    float kp3d_post[70 * 3];
    float kp2d_full[70 * 2], kp2d_crop[70 * 2], kp2d_dep[70];
    float pred_cam_t_world[3];

    int rc_total = 0, rc;
    for (int li = 0; li < NL; li++) {
        /* GPU: decoder layer li using x=tokens, x_pe=augment, ctx=ctx_in,
         *      ctx_pe=ctx_pe_tok → tokens_b. */
        rc = cuda_sam3d_body_debug_run_decoder_layer(
                ctx, li, tokens, ctx_in, augment, ctx_pe_tok,
                N_Q, N_C, tokens_b);
        if (rc != 0) {
            fprintf(stderr, "[cuda verify_decoder] decoder_layer %d rc=%d\n", li, rc);
            rc_total = 8; break;
        }
        { float *tmp = tokens; tokens = tokens_b; tokens_b = tmp; }

        if (li >= NL - 1) break;

        /* GPU: norm_final + heads. */
        rc = cuda_sam3d_body_debug_run_norm_and_heads(
                ctx, tokens, N_Q, tokens_n, pose_raw, cam_raw);
        if (rc != 0) {
            fprintf(stderr, "[cuda verify_decoder] norm_and_heads %d rc=%d\n", li, rc);
            rc_total = 9; break;
        }
        for (int i = 0; i < 519; i++) pose519[i] = pose_raw[i] + ip[i];
        for (int i = 0; i < 3;   i++) cam3[i]    = cam_raw[i]  + ic[i];

        /* CPU: decode_pose_raw + MHR forward + keypoints + camera_project. */
        if (sam3d_body_decode_pose_raw(m, pose519, /*enable_hand_model*/0,
                                       mp_buf, shape_buf, face_buf) != 0) {
            rc_total = 10; break;
        }
        if (sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)mhr,
                                   mp_buf, shape_buf, face_buf,
                                   1, 1, n_threads, mhr_scratch,
                                   verts_cm, gskel_cm) != 0) {
            rc_total = 11; break;
        }
        for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
        for (int j = 0; j < J; j++) {
            jc_m[j*3 + 0] = gskel_cm[(size_t)j*8 + 0] * 0.01f;
            jc_m[j*3 + 1] = gskel_cm[(size_t)j*8 + 1] * 0.01f;
            jc_m[j*3 + 2] = gskel_cm[(size_t)j*8 + 2] * 0.01f;
        }

        if (sam3d_body_keypoints_from_mesh(m, verts_m, jc_m, /*enable_hand_model*/0,
                                           n_threads, kp3d_post) != 0) {
            rc_total = 12; break;
        }
        if (sam3d_body_camera_project(kp3d_post, cam3, &B, K,
                                      kp2d_full, kp2d_crop, kp2d_dep,
                                      pred_cam_t_world) != 0) {
            rc_total = 13; break;
        }

        memcpy(dbg_kp3d  + (size_t)li * K * 3, kp3d_post,        K * 3 * sizeof(float));
        memcpy(dbg_kp2dc + (size_t)li * K * 2, kp2d_crop,        K * 2 * sizeof(float));
        memcpy(dbg_kp2dd + (size_t)li * K,     kp2d_dep,         K     * sizeof(float));
        memcpy(dbg_camt  + (size_t)li * 3,     pred_cam_t_world, 3     * sizeof(float));

        /* GPU: kp_token_update. */
        rc = cuda_sam3d_body_debug_run_kp_token_update(
                ctx, li, img_emb, H, W,
                kp2d_crop, kp2d_dep, kp3d_post,
                N_Q, tokens, augment);
        if (rc != 0) {
            fprintf(stderr, "[cuda verify_decoder] kp_token_update %d rc=%d\n", li, rc);
            rc_total = 14; break;
        }
    }

    if (rc_total == 0) {
        /* Final norm + heads (post-loop). */
        rc = cuda_sam3d_body_debug_run_norm_and_heads(
                ctx, tokens, N_Q, tokens_n, pose_raw, cam_raw);
        if (rc != 0) {
            fprintf(stderr, "[cuda verify_decoder] final norm_and_heads rc=%d\n", rc);
            rc_total = 9;
        }
    }

    float final_kp3d[70 * 3], final_kp2dc[70 * 2], final_kp2dd[70];
    float final_camt[3];
    if (rc_total == 0) {
        for (int i = 0; i < 519; i++) pose519[i] = pose_raw[i] + ip[i];
        for (int i = 0; i < 3;   i++) cam3[i]    = cam_raw[i]  + ic[i];

        if (sam3d_body_decode_pose_raw(m, pose519, /*enable_hand_model*/0,
                                       mp_buf, shape_buf, face_buf) != 0) {
            rc_total = 10;
        } else if (sam3d_body_mhr_forward((const sam3d_body_mhr_assets *)mhr,
                                          mp_buf, shape_buf, face_buf,
                                          1, 1, n_threads, mhr_scratch,
                                          verts_cm, gskel_cm) != 0) {
            rc_total = 11;
        } else {
            for (int i = 0; i < V * 3; i++) verts_m[i] = verts_cm[i] * 0.01f;
            for (int j = 0; j < J; j++) {
                jc_m[j*3 + 0] = gskel_cm[(size_t)j*8 + 0] * 0.01f;
                jc_m[j*3 + 1] = gskel_cm[(size_t)j*8 + 1] * 0.01f;
                jc_m[j*3 + 2] = gskel_cm[(size_t)j*8 + 2] * 0.01f;
            }
            if (sam3d_body_keypoints_from_mesh(m, verts_m, jc_m, 0,
                                               n_threads, final_kp3d) != 0) {
                rc_total = 12;
            } else if (sam3d_body_camera_project(final_kp3d, cam3, &B, K,
                                                 kp2d_full, final_kp2dc,
                                                 final_kp2dd, final_camt) != 0) {
                rc_total = 13;
            }
        }
    }

    /* Diff. */
    if (rc_total == 0) {
        /* Per-layer (0..NL-2). */
        int rc_d = 0;
        for (int li = 0; li < NL - 1; li++) {
            char nm[128], lbl[96];
            snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_3d", li);
            float *r_kp3d = load_or_die(refdir, nm);
            snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_cropped", li);
            float *r_kp2dc = load_or_die(refdir, nm);
            snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_depth", li);
            float *r_kp2dd = load_or_die(refdir, nm);
            snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_cam_t", li);
            float *r_camt = load_or_die(refdir, nm);
            if (!r_kp3d || !r_kp2dc || !r_kp2dd || !r_camt) {
                free(r_kp3d); free(r_kp2dc); free(r_kp2dd); free(r_camt);
                rc_d = 1;
                continue;
            }
            snprintf(lbl, sizeof(lbl), "layer%d kp3d (70,3)", li);
            rc_d |= diff_pair(lbl, dbg_kp3d + (size_t)li*K*3, r_kp3d, K*3, threshold);
            snprintf(lbl, sizeof(lbl), "layer%d kp2d_crop (70,2)", li);
            rc_d |= diff_pair(lbl, dbg_kp2dc + (size_t)li*K*2, r_kp2dc, K*2, threshold);
            snprintf(lbl, sizeof(lbl), "layer%d kp2d_depth (70,)", li);
            rc_d |= diff_pair(lbl, dbg_kp2dd + (size_t)li*K, r_kp2dd, K, threshold);
            snprintf(lbl, sizeof(lbl), "layer%d cam_t (3,)", li);
            rc_d |= diff_pair(lbl, dbg_camt + (size_t)li*3, r_camt, 3, threshold);
            free(r_kp3d); free(r_kp2dc); free(r_kp2dd); free(r_camt);
        }
        /* Final. */
        float *r_pose  = load_or_die(refdir, "head_pose_proj_raw");
        float *r_cam   = load_or_die(refdir, "head_camera_proj_raw");
        float *r_kp3df = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_3d");
        float *r_kp2dcf= load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_cropped");
        float *r_kp2ddf= load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_depth");
        float *r_camtf = load_or_die(refdir, "decoder_pose_layer5__pred_cam_t");
        if (r_pose && r_cam && r_kp3df && r_kp2dcf && r_kp2ddf && r_camtf) {
            rc_d |= diff_pair("head_pose_proj_raw (519,)",     pose_raw, r_pose, 519, threshold);
            rc_d |= diff_pair("head_camera_proj_raw (3,)",     cam_raw,  r_cam,  3,   threshold);
            rc_d |= diff_pair("final pred_keypoints_3d (70,3)",
                              final_kp3d, r_kp3df, K * 3, threshold);
            rc_d |= diff_pair("final pred_keypoints_2d_cropped (70,2)",
                              final_kp2dc, r_kp2dcf, K * 2, threshold);
            rc_d |= diff_pair("final pred_keypoints_2d_depth (70,)",
                              final_kp2dd, r_kp2ddf, K, threshold);
            rc_d |= diff_pair("final pred_cam_t (3,)",
                              final_camt, r_camtf, 3, threshold);
        } else {
            fprintf(stderr,
                    "[cuda verify_decoder] missing one or more final refs FAIL\n");
            rc_d = 1;
        }
        free(r_pose); free(r_cam); free(r_kp3df);
        free(r_kp2dcf); free(r_kp2ddf); free(r_camtf);
        rc_total = rc_d;
    }

    free(img_emb); free(init_x); free(init_xpe); free(ctx_pe_tok); free(ctx_in);
    free(tokens); free(tokens_b); free(augment); free(tokens_n);
    free(mhr_scratch); free(verts_cm); free(gskel_cm); free(verts_m); free(jc_m);
    free(dbg_kp3d); free(dbg_kp2dc); free(dbg_kp2dd); free(dbg_camt);
    cuda_sam3d_body_destroy(ctx);
    sam3d_body_mhr_free(mhr); sam3d_body_decoder_free(m);
    return rc_total;
}
