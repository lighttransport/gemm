/* CLI smoke test for HIP SAM 3 (Phase 1 — patch_embed + pos + pre-LN).
 * Runs the implemented stages and prints ours[0][:8] of the ViT embed.
 */
#include "hip_sam3_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <sam3.model.safetensors> <image.jpg>\n", argv[0]);
        return 1;
    }
    const char *ckpt = argv[1];
    const char *img_path = argv[2];

    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "failed to load %s\n", img_path); return 2; }

    hip_sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1008,
                             .device_ordinal = 0, .verbose = 1 };
    hip_sam3_ctx *ctx = hip_sam3_create(&cfg);
    if (!ctx) { stbi_image_free(rgb); return 3; }

    if (hip_sam3_set_image(ctx, rgb, H, W)) {
        hip_sam3_destroy(ctx); stbi_image_free(rgb); return 4;
    }

    int n_tok, D;
    float *emb = (float *)malloc((size_t)5184 * 1024 * sizeof(float));
    hip_sam3_get_vit_embed(ctx, emb, &n_tok, &D);
    fprintf(stderr, "vit_embed (%d, %d), tok0[:8]:", n_tok, D);
    for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", emb[i]);
    fprintf(stderr, "\n");

    free(emb);
    hip_sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
