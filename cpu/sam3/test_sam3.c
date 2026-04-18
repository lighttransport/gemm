/* CLI entry for SAM 3 CPU runner. */
#include "sam3_runner.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *p)
{
    fprintf(stderr,
        "Usage: %s <ckpt.pt> <image> --phrase \"a cat\" [-o out.npy] [-t N]\n", p);
}

int main(int argc, char **argv)
{
    if (argc < 5) { usage(argv[0]); return 1; }
    const char *ckpt = argv[1];
    const char *img_path = argv[2];
    const char *phrase = NULL;
    const char *out_path = "mask.npy";
    int threads = 4;

    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--phrase") && i + 1 < argc) phrase = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (!phrase) { fprintf(stderr, "--phrase required\n"); return 1; }

    int H, W, C;
    unsigned char *rgb = stbi_load(img_path, &W, &H, &C, 3);
    if (!rgb) { fprintf(stderr, "failed to load %s\n", img_path); return 2; }

    sam3_config cfg = { .ckpt_path = ckpt, .image_size = 1024, .num_threads = threads };
    sam3_ctx *ctx = sam3_create(&cfg);

    int rc = sam3_set_image(ctx, rgb, H, W);
    if (rc != 0) { sam3_destroy(ctx); stbi_image_free(rgb); return 3; }

    const int MAX_MASKS = 32;
    float *masks = (float *)malloc((size_t)MAX_MASKS * H * W * sizeof(float));
    float scores[32] = {0};
    int n = sam3_predict_text(ctx, phrase, MAX_MASKS, masks, scores);
    fprintf(stderr, "emitted %d masks; would write to %s (stub)\n", n, out_path);

    free(masks);
    sam3_destroy(ctx);
    stbi_image_free(rgb);
    return 0;
}
