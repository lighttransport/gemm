/*
 * verify_decoder — diff our promptable decoder + MHR head (regressor)
 * output against the pytorch reference dump.
 *
 * Usage:
 *   verify_decoder --safetensors-dir <dir> --refdir /tmp/sam3d_body_ref \
 *                  [--use-ref-inputs] [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   dinov3_tokens.npy  — upstream stage output (for --use-ref-inputs)
 *   mhr_params.npy     — decoder/MHR-head regression output (N floats)
 *
 * Legacy compatibility binary. The decoder is now covered by
 * verify_decoder_full and verify_decoder_forward; this binary fails
 * explicitly so stale scaffolds cannot hide missing coverage.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    float threshold = 1e-3f;
    int n_threads = 1, verbose = 0, use_ref_inputs = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--use-ref-inputs")) use_ref_inputs = 1;
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--use-ref-inputs] [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    (void)sft_dir;
    (void)refdir;
    (void)threshold;
    (void)n_threads;
    (void)verbose;
    (void)use_ref_inputs;
    fprintf(stderr,
            "[verify_decoder] obsolete scaffold. Use verify_decoder_full "
            "or verify_decoder_forward for maintained decoder coverage.\n");
    return 2;
}
