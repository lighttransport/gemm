/*
 * test_dinov2_gpu_upload — Phase 1b.0 scaffold microbench.
 *
 * Loads sam3d_dinov2.safetensors via cs3d_dinov2_gpu_load, then for a
 * sampled set of tensors does a host_ref vs D2H_readback bit-exact
 * diff. Goal: prove the upload path is correct and freeing is clean
 * before any kernel consumes these buffers in 1b.1+.
 *
 * Usage:
 *   ./test_dinov2_gpu_upload --safetensors-dir /mnt/disk01/models/sam3d/safetensors [-v]
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define CUDA_SAM3D_DINOV2_GPU_IMPLEMENTATION
#include "cuda_sam3d_dinov2_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int diff_ptr_against_safetensors(st_context *st, const char *name,
                                        CUdeviceptr d, size_t expect_n,
                                        const char *label, int verbose)
{
    int i = safetensors_find(st, name);
    if (i < 0) { fprintf(stderr, "[%s] missing %s\n", label, name); return -1; }
    size_t nb = safetensors_nbytes(st, i);
    if (nb / 4 != expect_n) {
        fprintf(stderr, "[%s] %s size mismatch (%zu vs %zu)\n", label, name,
                nb / 4, expect_n);
        return -1;
    }
    const float *host = (const float *)safetensors_data(st, i);
    float *back = (float *)malloc(nb);
    if (!back) return -1;
    if (cuMemcpyDtoH(back, d, nb) != CUDA_SUCCESS) {
        fprintf(stderr, "[%s] D2H failed for %s\n", label, name);
        free(back); return -1;
    }
    /* sam3d_dinov2 is F32 throughout, so the round-trip must be bit-exact. */
    int mismatch = memcmp(host, back, nb);
    if (mismatch != 0 || verbose) {
        size_t n = nb / 4;
        float mx = 0.0f;
        size_t at = 0;
        for (size_t k = 0; k < n; k++) {
            float d_ = fabsf(host[k] - back[k]);
            if (d_ > mx) { mx = d_; at = k; }
        }
        fprintf(stderr, "[%s] %s n=%zu max_abs=%.4g (at %zu)%s\n",
                label, name, n, (double)mx, at,
                mismatch ? "  FAIL (memcmp != 0)" : "");
    }
    free(back);
    return mismatch ? -1 : 0;
}

int main(int argc, char **argv)
{
    const char *safetensors_dir = "/mnt/disk01/models/sam3d/safetensors";
    int verbose = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) safetensors_dir = argv[++i];
        else if (!strcmp(a, "-v"))                              verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 3;
    }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 3; }
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet failed\n"); return 3; }
    CUcontext cu_ctx = NULL;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "cuCtxCreate failed\n"); return 3;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors", safetensors_dir);

    cs3d_dinov2_gpu g;
    if (cs3d_dinov2_gpu_load(&g, path, verbose ? 1 : 0) != 0) {
        fprintf(stderr, "cs3d_dinov2_gpu_load failed\n");
        return 4;
    }

    /* Re-open safetensors host-side so we can diff a sampled set of
     * tensors against their device-resident copies. */
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "safetensors_open(host-side) failed\n");
        cs3d_dinov2_gpu_free(&g);
        return 4;
    }

    int dim = g.dim, ffn = g.ffn_hidden, ps = g.patch_size;
    int og = g.orig_grid, nr = g.n_register, nb = g.n_blocks;
    int fail = 0;

    fail |= (diff_ptr_against_safetensors(st, "cls_token", g.cls_token,
                (size_t)dim, "top", verbose) != 0);
    if (g.register_tokens && nr > 0)
        fail |= (diff_ptr_against_safetensors(st, "register_tokens", g.register_tokens,
                    (size_t)nr * dim, "top", verbose) != 0);
    fail |= (diff_ptr_against_safetensors(st, "pos_embed", g.pos_embed,
                (size_t)(1 + og * og) * dim, "top", verbose) != 0);
    fail |= (diff_ptr_against_safetensors(st, "patch_embed.proj.weight", g.patch_w,
                (size_t)dim * 3 * ps * ps, "top", verbose) != 0);
    fail |= (diff_ptr_against_safetensors(st, "patch_embed.proj.bias", g.patch_b,
                (size_t)dim, "top", verbose) != 0);
    fail |= (diff_ptr_against_safetensors(st, "norm.weight", g.norm_w,
                (size_t)dim, "top", verbose) != 0);
    fail |= (diff_ptr_against_safetensors(st, "norm.bias", g.norm_b,
                (size_t)dim, "top", verbose) != 0);

    /* Sample blocks 0, mid, last. */
    int sample[3] = { 0, nb / 2, nb - 1 };
    for (int s = 0; s < 3; s++) {
        int L = sample[s];
        if (L < 0 || L >= nb) continue;
        cs3d_dinov2_gpu_block *b = &g.blocks[L];
        char nm[160], lbl[16];
        snprintf(lbl, sizeof(lbl), "blk%d", L);
        #define D(suf, ptr, n_elt) do {                                  \
            snprintf(nm, sizeof(nm), "blocks.%d." suf, L);               \
            fail |= (diff_ptr_against_safetensors(st, nm, ptr,           \
                                                  (size_t)(n_elt),       \
                                                  lbl, verbose) != 0);   \
        } while (0)
        D("norm1.weight",     b->norm1_w, dim);
        D("attn.qkv.weight",  b->qkv_w,   3 * dim * dim);
        D("attn.proj.weight", b->proj_w,  dim * dim);
        D("ls1.gamma",        b->ls1,     dim);
        D("mlp.fc1.weight",   b->fc1_w,   ffn * dim);
        D("mlp.fc2.weight",   b->fc2_w,   dim * ffn);
        D("ls2.gamma",        b->ls2,     dim);
        #undef D
    }

    safetensors_close(st);

    fprintf(stderr,
        "[test_dinov2_gpu_upload] dim=%d ffn=%d blocks=%d n_register=%d n_tokens=%d  "
        "uploaded=%.1f MiB  %s\n",
        g.dim, g.ffn_hidden, g.n_blocks, g.n_register, g.n_tokens,
        (double)g.total_bytes / (1024.0 * 1024.0),
        fail ? "FAIL" : "OK");

    cs3d_dinov2_gpu_free(&g);
    cuCtxDestroy(cu_ctx);
    return fail ? 7 : 0;
}
