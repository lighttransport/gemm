/*
 * test_ssdit_gpu_load — Phase 2c.10 standalone verification.
 *
 * Loads the real SS Flow DiT safetensors checkpoint, uploads ALL
 * weights (top-level embedders + 5 latent_mappings + N blocks × 2
 * streams) to the device via cs3d_ssdit_gpu_load, then for a single
 * sampled tensor (sa_qkv_w of block 0, shape stream) downloads it back
 * and diffs against the dequantized host buffer to confirm the upload
 * round-trip is bit-exact.
 *
 * Usage:
 *   ./test_ssdit_gpu_load --safetensors-dir <DIR> [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"
#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION
#include "../../common/sam3d_ss_flow_dit.h"

#define CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION
#include "cuda_sam3d_ssdit_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(a, "-v"))                              verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if (!sft_dir) {
        fprintf(stderr, "usage: %s --safetensors-dir <DIR> [-v]\n", argv[0]);
        return 2;
    }
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_ss_dit.safetensors", sft_dir);
    sam3d_ss_flow_dit_model *m = sam3d_ss_flow_dit_load_safetensors(path);
    if (!m) { fprintf(stderr, "load %s failed\n", path); return 3; }
    fprintf(stderr,
            "ssdit cpu model: D=%d H=%d D_h=%d n_blocks=%d mlp_h=%d cond=%d shortcut=%d\n",
            m->dim, m->n_heads, m->head_dim, m->n_blocks, m->mlp_hidden,
            m->cond_channels, m->is_shortcut);

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) return 5;
    if (cuInit(0) != CUDA_SUCCESS) return 5;
    CUdevice dev = 0; if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return 5;
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) return 5;

    cs3d_ssdit_gpu g;
    if (cs3d_ssdit_gpu_load(&g, m, verbose) < 0) {
        fprintf(stderr, "cs3d_ssdit_gpu_load failed\n");
        return 7;
    }

    /* Round-trip check: pick block 0 / shape-stream sa_qkv_w. */
    const qtensor *t = &m->blocks[0].stream[SAM3D_SS_STREAM_SHAPE].sa_qkv_w;
    int n = qt_numel(t);
    float *h_ref = qt_dequant(t);
    if (!h_ref) { fprintf(stderr, "qt_dequant failed\n"); return 8; }
    float *h_dev = (float *)malloc((size_t)n * sizeof(float));
    cuMemcpyDtoH(h_dev,
                 g.blocks[0].stream[SAM3D_SS_STREAM_SHAPE].sa_qkv_w,
                 (size_t)n * sizeof(float));
    cuCtxSynchronize();

    double sum = 0.0; float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(h_ref[i] - h_dev[i]);
        if (d > mx) mx = d;
        sum += d;
    }
    double mean = sum / (n > 0 ? n : 1);
    int ok = (mx == 0.0f);
    fprintf(stderr,
        "[test_ssdit_gpu_load] n_blocks=%d  total=%.1f MiB  "
        "round-trip blk0 shape sa_qkv_w (n=%d): max_abs=%.4g mean=%.4g  %s\n",
        g.n_blocks, (double)g.total_bytes / (1024.0 * 1024.0),
        n, (double)mx, mean, ok ? "OK" : "FAIL");

    free(h_ref); free(h_dev);
    cs3d_ssdit_gpu_free(&g);
    cuCtxDestroy(cu_ctx);
    sam3d_ss_flow_dit_free(m);
    return ok ? 0 : 9;
}
