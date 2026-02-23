/*
 * exr_writer.h - Thin C wrapper around tinyexr for writing multi-channel EXR
 *
 * Usage:
 *   #include "tinyexr.h"           // C declarations (implementation in tinyexr_impl.cc)
 *   #define EXR_WRITER_IMPLEMENTATION
 *   #include "exr_writer.h"
 *
 * Requires tinyexr.h to be included before this header for type/function declarations.
 */
#ifndef EXR_WRITER_H
#define EXR_WRITER_H

#include <stdio.h>
#include <string.h>

/* Forward declaration - must match cuda_da3_runner.h */
struct da3_full_result_tag;

/* Write depth + confidence as 2-channel float EXR */
static int write_exr_depth(const char *path, const float *depth,
                           const float *confidence, int w, int h);

/* Write all available modalities as named channels.
 * result must be a da3_full_result pointer. Declared as void* to avoid
 * requiring cuda_da3_runner.h in this header. */
static int write_exr_full(const char *path, const void *result_ptr,
                          int w, int h);

#ifdef EXR_WRITER_IMPLEMENTATION

static int write_exr_depth(const char *path, const float *depth,
                           const float *confidence, int w, int h) {
    int n_channels = confidence ? 2 : 1;

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = n_channels;
    image.width = w;
    image.height = h;

    /* tinyexr expects array of per-channel pointers */
    const float *channel_ptrs[2];
    /* EXR channels are sorted alphabetically by tinyexr: confidence, depth */
    if (confidence) {
        channel_ptrs[0] = confidence;  /* "confidence" */
        channel_ptrs[1] = depth;       /* "depth" */
    } else {
        channel_ptrs[0] = depth;
    }
    image.images = (unsigned char **)channel_ptrs;

    header.num_channels = n_channels;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * (size_t)n_channels);

    /* Channels must be alphabetically sorted for EXR */
    if (confidence) {
        strncpy(header.channels[0].name, "confidence", 255);
        strncpy(header.channels[1].name, "depth", 255);
    } else {
        strncpy(header.channels[0].name, "depth", 255);
    }

    header.pixel_types = (int *)malloc(sizeof(int) * (size_t)n_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * (size_t)n_channels);
    for (int i = 0; i < n_channels; i++) {
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    const char *err = NULL;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "EXR write error: %s\n", err ? err : "unknown");
        FreeEXRErrorMessage(err);
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret == TINYEXR_SUCCESS)
        fprintf(stderr, "Wrote %s (%dx%d, %d channels)\n", path, w, h, n_channels);
    return ret == TINYEXR_SUCCESS ? 0 : -1;
}

static int write_exr_full(const char *path, const void *result_ptr,
                          int w, int h) {
    /* We access the result fields via known offsets matching da3_full_result.
     * To avoid header dependency, we cast from void*. The caller must ensure
     * this is actually a da3_full_result*. */
    typedef struct {
        float *depth, *confidence;
        float *rays;
        float *ray_confidence;
        float *sky_seg;
        float pose[9];
        float *gaussians;
        float *metric_depth;
        int width, height;
        int has_pose, has_rays, has_gaussians, has_metric;
    } da3_fr;
    const da3_fr *r = (const da3_fr *)result_ptr;

    /* Count channels */
    int n_ch = 0;
    /* depth + confidence: always present */
    n_ch += 2;
    /* rays: 6 direction + 1 confidence = 7 */
    if (r->has_rays && r->rays) n_ch += 6;
    if (r->has_rays && r->ray_confidence) n_ch += 1;
    /* gaussians: 38 channels */
    if (r->has_gaussians && r->gaussians) n_ch += 38;

    /* Max channels we might write */
    #define MAX_EXR_CH 64
    if (n_ch > MAX_EXR_CH) return -1;

    const float *ch_ptrs[MAX_EXR_CH];
    char ch_names[MAX_EXR_CH][256];
    int ci = 0;

    /* Add channels - we'll sort alphabetically after */
    /* Confidence */
    if (r->confidence) {
        snprintf(ch_names[ci], 256, "confidence");
        ch_ptrs[ci] = r->confidence;
        ci++;
    }
    /* Depth */
    snprintf(ch_names[ci], 256, "depth");
    ch_ptrs[ci] = r->depth;
    ci++;

    /* Gaussians: gs.* */
    if (r->has_gaussians && r->gaussians) {
        int npix = w * h;
        const char *gs_ch_names[38] = {
            "gs.offset_x", "gs.offset_y",
            "gs.scale_0", "gs.scale_1", "gs.scale_2",
            "gs.rot_0", "gs.rot_1", "gs.rot_2", "gs.rot_3",
            "gs.sh_00", "gs.sh_01", "gs.sh_02", "gs.sh_03",
            "gs.sh_04", "gs.sh_05", "gs.sh_06", "gs.sh_07",
            "gs.sh_08", "gs.sh_09", "gs.sh_10", "gs.sh_11",
            "gs.sh_12", "gs.sh_13", "gs.sh_14", "gs.sh_15",
            "gs.sh_16", "gs.sh_17", "gs.sh_18", "gs.sh_19",
            "gs.sh_20", "gs.sh_21", "gs.sh_22", "gs.sh_23",
            "gs.sh_24", "gs.sh_25", "gs.sh_26",
            "gs.off_depth", "gs.opacity"
        };
        for (int c = 0; c < 38; c++) {
            snprintf(ch_names[ci], 256, "%s", gs_ch_names[c]);
            ch_ptrs[ci] = r->gaussians + (size_t)c * npix;
            ci++;
        }
    }

    /* Rays: ray.* */
    if (r->has_rays && r->rays) {
        int npix = w * h;
        const char *ray_dirs[6] = {"ray.X", "ray.Y", "ray.Z", "ray.dX", "ray.dY", "ray.dZ"};
        for (int c = 0; c < 6; c++) {
            snprintf(ch_names[ci], 256, "%s", ray_dirs[c]);
            ch_ptrs[ci] = r->rays + (size_t)c * npix;
            ci++;
        }
        if (r->ray_confidence) {
            snprintf(ch_names[ci], 256, "ray.confidence");
            ch_ptrs[ci] = r->ray_confidence;
            ci++;
        }
    }

    n_ch = ci;

    /* Sort channels alphabetically (EXR requirement) using simple insertion sort */
    for (int i = 1; i < n_ch; i++) {
        char tmp_name[256];
        const float *tmp_ptr = ch_ptrs[i];
        memcpy(tmp_name, ch_names[i], 256);
        int j = i - 1;
        while (j >= 0 && strcmp(ch_names[j], tmp_name) > 0) {
            memcpy(ch_names[j + 1], ch_names[j], 256);
            ch_ptrs[j + 1] = ch_ptrs[j];
            j--;
        }
        memcpy(ch_names[j + 1], tmp_name, 256);
        ch_ptrs[j + 1] = tmp_ptr;
    }

    /* Build EXR header + image */
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = n_ch;
    image.width = w;
    image.height = h;
    image.images = (unsigned char **)ch_ptrs;

    header.num_channels = n_ch;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * (size_t)n_ch);
    header.pixel_types = (int *)malloc(sizeof(int) * (size_t)n_ch);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * (size_t)n_ch);

    for (int i = 0; i < n_ch; i++) {
        strncpy(header.channels[i].name, ch_names[i], 255);
        header.channels[i].name[255] = '\0';
        header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    /* Store pose as EXR custom attributes if available */
    if (r->has_pose) {
        char pose_str[256];
        snprintf(pose_str, sizeof(pose_str),
                 "tx=%.6f ty=%.6f tz=%.6f qx=%.6f qy=%.6f qz=%.6f qw=%.6f fov_x=%.6f fov_y=%.6f",
                 r->pose[0], r->pose[1], r->pose[2],
                 r->pose[3], r->pose[4], r->pose[5], r->pose[6],
                 r->pose[7], r->pose[8]);
        header.num_custom_attributes = 1;
        header.custom_attributes = (EXRAttribute *)malloc(sizeof(EXRAttribute));
        strncpy(header.custom_attributes[0].name, "da3_pose", 255);
        strncpy(header.custom_attributes[0].type, "string", 255);
        header.custom_attributes[0].size = (int)strlen(pose_str);
        header.custom_attributes[0].value = (unsigned char *)malloc((size_t)header.custom_attributes[0].size);
        memcpy(header.custom_attributes[0].value, pose_str, (size_t)header.custom_attributes[0].size);
    }

    const char *err = NULL;
    int ret = SaveEXRImageToFile(&image, &header, path, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "EXR write error: %s\n", err ? err : "unknown");
        FreeEXRErrorMessage(err);
    }

    if (r->has_pose && header.custom_attributes) {
        free(header.custom_attributes[0].value);
        free(header.custom_attributes);
    }
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret == TINYEXR_SUCCESS)
        fprintf(stderr, "Wrote %s (%dx%d, %d channels)\n", path, w, h, n_ch);
    return ret == TINYEXR_SUCCESS ? 0 : -1;
    #undef MAX_EXR_CH
}

#endif /* EXR_WRITER_IMPLEMENTATION */

#endif /* EXR_WRITER_H */
