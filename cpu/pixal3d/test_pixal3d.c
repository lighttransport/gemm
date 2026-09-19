/* Analytic smoke checks for the public C API; no weights or GPU are required. */
#include "../../common/pixal3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                                             \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #condition);                     \
            return 1;                                                                                \
        }                                                                                            \
    } while (0)

static int near(float a, float b) { return isfinite(a) && fabsf(a - b) <= 1e-6f; }

int main(void) {
    float distance;
    const float half_pi = 1.5707963267948966f;
    CHECK(pixal3d_camera_distance(half_pi, 2, &distance) == 0 && near(distance, .25f));
    CHECK(pixal3d_camera_distance(NAN, 1, &distance) == -1);
    CHECK(pixal3d_camera_distance(half_pi, 0, &distance) == -1);

    /* Principal point and positive X/Y offsets, including the half-pixel shift. */
    const int32_t coords[] = {0, 1, 1, 1, 0, 2, 1, 1, 0, 1, 2, 1};
    const pixal3d_camera camera = {half_pi, 1, 1};
    float projected[6];
    const float expected_projection[] = {.125f, .125f, .625f, .125f, .125f, -.375f};
    CHECK(pixal3d_project(coords, 3, 3, 8, &camera, projected) == 0);
    for (size_t i = 0; i < 6; ++i)
        CHECK(near(projected[i], expected_projection[i]));
    const float front_view[] = {1, 0, 0, 0, 0, 0, -1, -1, 0, 1, 0, 0, 0, 0, 0, 1};
    float matrix_projected[6];
    CHECK(pixal3d_project_matrix(coords, 3, 3, 8, half_pi, 1, front_view, matrix_projected) == 0);
    for (size_t i = 0; i < 6; ++i)
        CHECK(near(matrix_projected[i], expected_projection[i]));
    CHECK(pixal3d_project_matrix(coords, 3, 3, 8, half_pi, 1, (float[16]){0}, matrix_projected) == -1);

    /* Pixel center, four-pixel mean and opposite border clamps in HWC order. */
    const float features[] = {0, 10, 2, 12, 4, 14, 6, 16};
    const float queries[] = {-.5f, -.5f, 0, 0, 10, -10, -10, 10};
    const float expected_samples[] = {0, 10, 3, 13, 2, 12, 4, 14};
    float sampled[8];
    CHECK(pixal3d_sample_features(features, 2, 2, 2, queries, 4, sampled) == 0);
    for (size_t i = 0; i < 8; ++i)
        CHECK(near(sampled[i], expected_samples[i]));

    /* At this scale coordinates map to 0.5, 1.5 and 2.5: ties round to 0, 2, 2. */
    const int32_t coarse[] = {0, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0};
    const int32_t expected_coords[] = {0, 0, 0, 0, 0, 2, 2, 2};
    int32_t unique[16];
    CHECK(pixal3d_cascade_coords(coarse, 4, 3, 64, unique) == 2);
    CHECK(memcmp(unique, expected_coords, sizeof(expected_coords)) == 0);

    const float positive[] = {2, -2}, negative[] = {.5f, -.5f};
    float sample[] = {1, -1};
    pixal3d_euler_cfg(sample, positive, negative, 2, 1, .75f, 2, 0, 0);
    CHECK(near(sample[0], .125f) && near(sample[1], -.125f));
    /* x0 rescaling differs from velocity rescaling for this nonzero sample. */
    const float rescale_positive[] = {1, -1}, zero[] = {0, 0};
    float rescaled[] = {1, 1};
    pixal3d_euler_cfg(rescaled, rescale_positive, zero, 2, 1, .5f, 2, 1, 0);
    CHECK(near(rescaled[0], .25f) && near(rescaled[1], 1.25f));

    pixal3d_gpu_options gpu;
    pixal3d_default_gpu_options(&gpu);
    CHECK(gpu.struct_size == sizeof(gpu) && gpu.version == 1);
    CHECK(gpu.execution == PIXAL3D_GPU_LEGACY && gpu.kernels == PIXAL3D_KERNEL_AUTO);
    CHECK(gpu.profile_json == NULL && pixal3d_configure_gpu(NULL, &gpu) == -1);
    pixal3d_default_gpu_options(NULL);
    pixal3d_options options;
    pixal3d_default_options(&options);
    CHECK(options.backend == PIXAL3D_CPU && options.texture_size == 4096);
    CHECK(options.vram_budget_mib == 14336 && options.decimation_target == 1000000);
    CHECK(pixal3d_create(NULL) == NULL && strlen(pixal3d_last_error(NULL)) > 0);
    CHECK(pixal3d_generate(NULL, NULL, NULL, NULL) == -1);
    CHECK(pixal3d_generate_multiview(NULL, NULL, 0, NULL) == -1);
    pixal3d_result result = {0};
    result.vertices = malloc(3 * sizeof(float));
    CHECK(result.vertices != NULL);
    result.vertex_count = 1;
    pixal3d_result_free(&result);
    CHECK(result.vertices == NULL && result.vertex_count == 0);
    pixal3d_result_free(&result);
    pixal3d_destroy(NULL);
    puts("Pixal3D native C API and analytic math: PASS");
    return 0;
}
