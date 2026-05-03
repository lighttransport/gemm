/*
 * verify_mhr — diff our MHR skinning (MHR params → 3D vertices)
 * against the pytorch reference dump.
 *
 * Usage:
 *   verify_mhr --mhr-assets <dir> --refdir /tmp/sam3d_body_ref \
 *              [--use-ref-inputs] [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   mhr_params.npy     — (N,) regressor output
 *   out_vertices.npy   — (V, 3) skinned mesh in root-relative frame
 *   out_keypoints_3d.npy — (K, 3) joints in same frame
 *
 * Legacy compatibility binary. MHR is now either verified per-stage by
 * verify_mhr_stages or exercised inline by decoder/full end-to-end
 * checks; this binary fails explicitly so stale scaffolds cannot hide
 * missing coverage.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL, *mhr_dir = NULL;
    float threshold = 1e-3f;
    int n_threads = 1, verbose = 0, use_ref_inputs = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--use-ref-inputs"))                use_ref_inputs = 1;
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!mhr_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --mhr-assets <dir> --refdir <dir> "
                "[--use-ref-inputs] [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    (void)sft_dir;
    (void)refdir;
    (void)mhr_dir;
    (void)threshold;
    (void)n_threads;
    (void)verbose;
    (void)use_ref_inputs;
    fprintf(stderr,
            "[verify_mhr] obsolete scaffold. Use verify_mhr_stages for "
            "standalone MHR stage coverage, or verify_decoder_full / "
            "verify_end_to_end for inline MHR coverage.\n");
    return 2;
}
