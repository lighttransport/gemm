/* Native FLUX 3 Image Klein flow-map DiT for RDNA4.
 * Architecture follows BFL flux3/dit.py (Apache-2.0), revision documented in README.
 * SPDX-License-Identifier: MIT */
#include "hip_flux3_runner.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../rocew.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#pragma GCC diagnostic pop
#include "hip_flux3_kernels.inc"
#include <limits.h>

enum { H = 3072, FF = 9216, TD = 20480, MAX_OWNED = 768 };
typedef struct {
    void *data;
    float *scale;
    int rows, cols, fp8;
    char name[192];
} weight;
typedef struct {
    weight q, k, v, attn, mi, mo;
    float *qn, *kn;
} block;
struct hip_flux3_runner {
    int device, verbose, loaded;
    const char *trace_dir;
    size_t reserve;
    hipModule_t module;
    hipFunction_t prepare, linear, round, silu, add_ref, lnmod, normrope, swiglu, residual,
        attention;
    void *owned[MAX_OWNED];
    int nowned;
    weight input[3], time[2], ref[2], early[3], single[3], final_mod, final_out;
    block early_block[3][5], single_block[28];
    void *workspace;
    size_t capacity;
    float *x, *ln, *q, *k, *v, *att, *ap, *mp, *mlp, *sg, *prep, *xs, *in, *ids;
    float *vec, *refvec, *embed, *tmpvec, *emod, *smod, *fmod;
    int *mode;
};
static float bf_host(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
    memcpy(&x, &u, 4);
    return x;
}
static int room(hip_flux3_runner *r, size_t bytes) {
    size_t avail, total;
    HIP_CHECK(hipMemGetInfo(&avail, &total));
    if (avail < r->reserve || bytes > avail - r->reserve) {
        fprintf(stderr, "flux3: need %.1f MiB plus %.1f MiB reserve; %.1f MiB free\n",
                bytes / 1048576., r->reserve / 1048576., avail / 1048576.);
        return -1;
    }
    return 0;
}
static int launch(hipFunction_t f, unsigned gx, unsigned gy, unsigned threads, void **args) {
    HIP_CHECK(hipModuleLaunchKernel(f, gx, gy, 1, threads, 1, 1, 0, NULL, args, NULL));
    return 0;
}
#define TRY(x)                                                                                     \
    do {                                                                                           \
        if ((x) < 0)                                                                               \
            return -1;                                                                             \
    } while (0)
static void *upload(hip_flux3_runner *r, const void *p, size_t n) {
    if (r->nowned >= MAX_OWNED)
        return NULL;
    void *d = NULL;
    if (hipMalloc(&d, n) != hipSuccess)
        return NULL;
    if (hipMemcpy(d, p, n, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(d);
        return NULL;
    }
    r->owned[r->nowned++] = d;
    return d;
}
static int tensor(hip_flux3_runner *r, st_context *s, const char *name, int rows, int cols,
                  weight *w) {
    int i = safetensors_find(s, name);
    if (i < 0 || safetensors_ndims(s, i) != 2 || safetensors_shape(s, i)[0] != (unsigned)rows ||
        safetensors_shape(s, i)[1] != (unsigned)cols) {
        fprintf(stderr, "flux3: missing/wrong tensor %s [%d,%d]\n", name, rows, cols);
        return -1;
    }
    const char *d = safetensors_dtype(s, i);
    snprintf(w->name, sizeof(w->name), "%.*s", (int)(strlen(name) - 7), name);
    w->rows = rows;
    w->cols = cols;
    w->fp8 = !strcmp(d, "F8_E4M3");
    if (!w->fp8 && strcmp(d, "BF16")) {
        fprintf(stderr, "flux3: unsupported dtype %s for %s (expected BF16 or F8_E4M3)\n", d, name);
        return -1;
    }
    w->data = upload(r, safetensors_data(s, i), safetensors_nbytes(s, i));
    if (!w->data)
        return -1;
    if (w->fp8) {
        char key[256];
        snprintf(key, sizeof(key), "%.*sweight_scale", (int)(strlen(name) - 6), name);
        int j = safetensors_find(s, key);
        if (j < 0 || strcmp(safetensors_dtype(s, j), "F32") ||
            safetensors_nbytes(s, j) != (size_t)rows * 4) {
            fprintf(stderr, "flux3: missing per-output-row F32 scales %s\n", key);
            return -1;
        }
        w->scale = upload(r, safetensors_data(s, j), (size_t)rows * 4);
        if (!w->scale)
            return -1;
    }
    return 0;
}
static int norm(hip_flux3_runner *r, st_context *s, const char *name, float **out) {
    int i = safetensors_find(s, name);
    float a[128];
    if (i < 0 || safetensors_ndims(s, i) != 1 || safetensors_shape(s, i)[0] != 128)
        return -1;
    const char *d = safetensors_dtype(s, i);
    void *p = safetensors_data(s, i);
    for (int j = 0; j < 128; j++) {
        if (!strcmp(d, "BF16")) {
            uint32_t v = (uint32_t)((uint16_t *)p)[j] << 16;
            memcpy(a + j, &v, 4);
        } else if (!strcmp(d, "F32"))
            a[j] = ((float *)p)[j];
        else
            return -1;
    }
    *out = upload(r, a, sizeof(a));
    return *out ? 0 : -1;
}
static int load_block(hip_flux3_runner *r, st_context *s, const char *prefix, block *b) {
    char key[256];
#define W(field, suffix, rows, cols)                                                               \
    do {                                                                                           \
        snprintf(key, sizeof(key), "%s.%s", prefix, suffix);                                       \
        TRY(tensor(r, s, key, rows, cols, &b->field));                                             \
    } while (0)
    W(q, "q_proj.weight", H, H);
    W(k, "k_proj.weight", H, H);
    W(v, "v_proj.weight", H, H);
    W(attn, "attn_out.weight", H, H);
    W(mi, "mlp_in.weight", 2 * FF, H);
    W(mo, "mlp_out.weight", H, FF);
#undef W
    snprintf(key, sizeof(key), "%s.norm.query_norm.scale", prefix);
    TRY(norm(r, s, key, &b->qn));
    snprintf(key, sizeof(key), "%s.norm.key_norm.scale", prefix);
    TRY(norm(r, s, key, &b->kn));
    return 0;
}
hip_flux3_runner *hip_flux3_init(int device, size_t reserve_mib, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS)
        return NULL;
    if (hipSetDevice(device) != hipSuccess)
        return NULL;
    const char *arch = rocewGetRDNA4ArchString(device);
    if (!arch || strncmp(arch, "gfx12", 5)) {
        fprintf(stderr, "flux3: BF16 WMMA requires gfx12\n");
        return NULL;
    }
    hip_flux3_runner *r = calloc(1, sizeof(*r));
    if (!r)
        return NULL;
    r->device = device;
    r->verbose = verbose;
    r->reserve = reserve_mib * 1048576;
    if (hip_compile_kernels_ex(&r->module, device, flux3_kernels, "flux3.hip", verbose, "hip_flux3",
                               1) < 0)
        goto fail;
#define FN(field, name)                                                                            \
    if (hipModuleGetFunction(&r->field, r->module, name) != hipSuccess)                            \
    goto fail
    FN(prepare, "prepare");
    FN(linear, "linear");
    FN(round, "round_bf");
    FN(silu, "silu");
    FN(add_ref, "add_ref");
    FN(lnmod, "lnmod");
    FN(normrope, "normrope");
    FN(swiglu, "swiglu");
    FN(residual, "residual");
    FN(attention, "attention");
#undef FN
    return r;
fail:
    hip_flux3_free(r);
    return NULL;
}
int hip_flux3_load(hip_flux3_runner *r, const char *path) {
    if (!r || !path)
        return -1;
    HIP_CHECK(hipSetDevice(r->device));
    if (r->loaded || r->nowned) {
        fprintf(stderr, "flux3: create a fresh runner before loading another checkpoint\n");
        return -1;
    }
    st_context *s = safetensors_open(path);
    if (!s)
        return -1;
    size_t bytes = 0;
    for (int i = 0; i < s->n_tensors; i++)
        bytes += safetensors_nbytes(s, i);
    if (room(r, bytes) < 0) {
        safetensors_close(s);
        return -1;
    }
    int rc = -1;
    char key[256];
    const char *modes[] = {"image", "image_cond", "txt"};
#define LOAD(name, rows, cols, dest)                                                               \
    if (tensor(r, s, name, rows, cols, dest) < 0)                                                  \
    goto done
    LOAD("emb_in.image.weight", H, 128, &r->input[0]);
    LOAD("emb_in.image_cond.weight", H, 128, &r->input[1]);
    LOAD("txt_in.weight", H, TD, &r->input[2]);
    LOAD("time_in.in_layer.weight", H, 256, &r->time[0]);
    LOAD("time_in.out_layer.weight", H, H, &r->time[1]);
    LOAD("ref_time_in.in_layer.weight", H, 256, &r->ref[0]);
    LOAD("ref_time_in.out_layer.weight", H, H, &r->ref[1]);
    for (int m = 0; m < 3; m++) {
        snprintf(key, sizeof(key), "early_stream_modulations.%s.lin.weight", modes[m]);
        LOAD(key, 3 * H, H, &r->early[m]);
        snprintf(key, sizeof(key), "single_stream_modulations.%s.lin.weight", modes[m]);
        LOAD(key, 3 * H, H, &r->single[m]);
        for (int d = 0; d < 5; d++) {
            if (m == 2)
                snprintf(key, sizeof(key), "txt_mode_blocks.%d", d);
            else
                snprintf(key, sizeof(key), "content_mode_blocks.%s.%d", modes[m], d);
            if (load_block(r, s, key, &r->early_block[m][d]) < 0)
                goto done;
        }
    }
    for (int d = 0; d < 28; d++) {
        snprintf(key, sizeof(key), "single_blocks.%d", d);
        if (load_block(r, s, key, &r->single_block[d]) < 0)
            goto done;
    }
    LOAD("final_layer.image.adaLN_modulation.1.weight", 2 * H, H, &r->final_mod);
    LOAD("final_layer.image.linear.weight", 128, H, &r->final_out);
#undef LOAD
    r->loaded = 1;
    rc = 0;
    if (r->verbose)
        fprintf(stderr, "flux3: loaded %.2f GiB weights, 5 per-mode + 28 shared blocks\n",
                bytes / 1073741824.);
done:
    safetensors_close(s);
    return rc;
}
static int workspace(hip_flux3_runner *r, int n) {
    size_t floats = (size_t)n * (8 * H + 2 * FF + FF + 2 * TD + 1 + 4) + 3 * H + H + 3 * 256 +
                    3 * H + 2 * 9 * H + 2 * H + 3;
    size_t bytes = floats * 4 + (size_t)n * 4;
    if (r->capacity >= bytes)
        return 0;
    if (r->workspace) {
        HIP_CHECK(hipFree(r->workspace));
        r->workspace = NULL;
        r->capacity = 0;
    }
    TRY(room(r, bytes));
    HIP_CHECK(hipMalloc(&r->workspace, bytes));
    r->capacity = bytes;
    float *p = r->workspace;
#define BUF(field, count)                                                                          \
    r->field = p;                                                                                  \
    p += (count)
    BUF(x, (size_t)n * H);
    BUF(ln, (size_t)n * H);
    BUF(q, (size_t)n * H);
    BUF(k, (size_t)n * H);
    BUF(v, (size_t)n * H);
    BUF(att, (size_t)n * H);
    BUF(ap, (size_t)n * H);
    BUF(mp, (size_t)n * H);
    BUF(mlp, (size_t)n * 2 * FF);
    BUF(sg, (size_t)n * FF);
    BUF(prep, (size_t)n * TD);
    BUF(xs, n + 3);
    BUF(in, (size_t)n * TD);
    BUF(ids, (size_t)n * 4);
    BUF(vec, 3 * H);
    BUF(refvec, H);
    BUF(embed, 3 * 256);
    BUF(tmpvec, 3 * H);
    BUF(emod, 9 * H);
    BUF(smod, 9 * H);
    BUF(fmod, 2 * H);
#undef BUF
    r->mode = (int *)p;
    return 0;
}
void hip_flux3_set_trace(hip_flux3_runner *r, const char *directory) {
    if (r)
        r->trace_dir = directory;
}
static int trace(hip_flux3_runner *r, const char *name, const float *data, size_t n) {
    if (!r->trace_dir)
        return 0;
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s.f32", r->trace_dir, name);
    float *host = malloc(n * 4);
    if (!host)
        return -1;
    hipError_t status = hipMemcpy(host, data, n * 4, hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
        free(host);
        return -1;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        free(host);
        return -1;
    }
    int ok = fwrite(host, 4, n, f) == n;
    free(host);
    if (fclose(f))
        ok = 0;
    return ok ? 0 : -1;
}
static int linear_op(hip_flux3_runner *r, const weight *w, const float *x, float *y, int m) {
    int k = w->cols, n = w->rows, q = w->fp8;
    void *a[] = {&r->prep, &r->xs, &x, &k, &q};
    TRY(launch(r->prepare, m, 1, 256, a));
    void *wd = w->data;
    float *ws = w->scale;
    void *b[] = {&y, &wd, &ws, &r->prep, &r->xs, &n, &k, &m, &q};
    TRY(launch(r->linear, (n + 15) / 16, (m + 63) / 64, 128, b));
    TRY(trace(r, w->name, y, (size_t)m * n));
    char name[256];
    snprintf(name, sizeof(name), "%s.input", w->name);
    return trace(r, name, x, (size_t)m * k);
}
static int silu_op(hip_flux3_runner *r, float *y, const float *x, int n) {
    void *a[] = {&y, &x, &n};
    return launch(r->silu, (n + 255) / 256, 1, 256, a);
}
static int ln_op(hip_flux3_runner *r, float *y, const float *x, const float *mod, const int *mode,
                 int n) {
    void *a[] = {&y, &x, &mod, &mode, &n};
    return launch(r->lnmod, n, 1, 256, a);
}
static int block_op(hip_flux3_runner *r, const block *b, float *x, const float *ids,
                    const int *mode, const float *mod, int n) {
    TRY(ln_op(r, r->ln, x, mod, mode, n));
    TRY(linear_op(r, &b->q, r->ln, r->q, n));
    TRY(linear_op(r, &b->k, r->ln, r->k, n));
    TRY(linear_op(r, &b->v, r->ln, r->v, n));
    float *qn = b->qn, *kn = b->kn;
    void *a[] = {&r->q, &qn, &ids, &n};
    TRY(launch(r->normrope, n * 24, 1, 128, a));
    void *c[] = {&r->k, &kn, &ids, &n};
    TRY(launch(r->normrope, n * 24, 1, 128, c));
    char name[256];
    snprintf(name, sizeof(name), "%.*s.norm.q", (int)(strlen(b->q.name) - 7), b->q.name);
    TRY(trace(r, name, r->q, (size_t)n * H));
    snprintf(name, sizeof(name), "%.*s.norm.k", (int)(strlen(b->q.name) - 7), b->q.name);
    TRY(trace(r, name, r->k, (size_t)n * H));
    void *d[] = {&r->att, &r->q, &r->k, &r->v, &n};
    TRY(launch(r->attention, 24, (n + 3) / 4, 128, d));
    TRY(linear_op(r, &b->attn, r->att, r->ap, n));
    TRY(linear_op(r, &b->mi, r->ln, r->mlp, n));
    void *e[] = {&r->sg, &r->mlp, &n};
    TRY(launch(r->swiglu, ((size_t)n * FF + 255) / 256, 1, 256, e));
    TRY(linear_op(r, &b->mo, r->sg, r->mp, n));
    void *f[] = {&x, &r->ap, &r->mp, &mod, &mode, &n};
    return launch(r->residual, ((size_t)n * H + 255) / 256, 1, 256, f);
}
static void time_embed(float *out, float t) {
    float v = bf_host(1000.f * bf_host(t));
    for (int j = 0; j < 128; j++) {
        float a = v * expf(-logf(10000.f) * j / 128.f);
        out[j] = bf_host(cosf(a));
        out[j + 128] = bf_host(sinf(a));
    }
}
int hip_flux3_step(hip_flux3_runner *r, const float *image, const float *image_ids, int ni,
                   const float *condition, const float *condition_ids, int nc, const float *text,
                   const float *text_ids, int nt, float t, float target, float *out) {
    if (!r || !r->loaded || ni < 1 || nt < 1 || nc < 0 || ni > 16384 || nt > 4096 || nc > 16384 ||
        !image || !image_ids || !text || !text_ids || !out ||
        (nc && (!condition || !condition_ids)) || !isfinite(t) || !isfinite(target) || t < 0 ||
        t > 1 || target < 0 || target > 1)
        return -1;
    HIP_CHECK(hipSetDevice(r->device));
    int n = ni + nc + nt, count[] = {ni, nc, nt}, start[] = {0, ni, ni + nc};
    TRY(workspace(r, n));
    float emb[3 * 256];
    time_embed(emb, t);
    time_embed(emb + 256, 0);
    time_embed(emb + 512, 0);
    HIP_CHECK(hipMemcpy(r->embed, emb, sizeof(emb), hipMemcpyHostToDevice));
    TRY(linear_op(r, &r->time[0], r->embed, r->tmpvec, 3));
    TRY(silu_op(r, r->tmpvec, r->tmpvec, 3 * H));
    TRY(linear_op(r, &r->time[1], r->tmpvec, r->vec, 3));
    time_embed(emb, target);
    HIP_CHECK(hipMemcpy(r->embed, emb, 256 * 4, hipMemcpyHostToDevice));
    TRY(linear_op(r, &r->ref[0], r->embed, r->tmpvec, 1));
    TRY(silu_op(r, r->tmpvec, r->tmpvec, H));
    TRY(linear_op(r, &r->ref[1], r->tmpvec, r->refvec, 1));
    int vn = 3 * H;
    void *a[] = {&r->vec, &r->refvec, &vn};
    TRY(launch(r->add_ref, (vn + 255) / 256, 1, 256, a));
    TRY(silu_op(r, r->tmpvec, r->vec, 3 * H));
    const float *inputs[] = {image, condition, text},
                *allids[] = {image_ids, condition_ids, text_ids};
    int *hostmode = malloc((size_t)n * 4);
    if (!hostmode)
        return -1;
    for (int m = 0; m < 3; m++)
        for (int j = 0; j < count[m]; j++)
            hostmode[start[m] + j] = m;
    hipError_t copy_status = hipMemcpy(r->mode, hostmode, (size_t)n * 4, hipMemcpyHostToDevice);
    free(hostmode);
    HIP_CHECK(copy_status);
    for (int m = 0; m < 3; m++) {
        if (!count[m])
            continue;
        char trace_name[192];
        const char *mnames[] = {"image", "image_cond", "txt"};
        snprintf(trace_name, sizeof(trace_name), "early_stream_modulations.%s.input", mnames[m]);
        TRY(trace(r, trace_name, r->vec + m * H, H));
        TRY(linear_op(r, &r->early[m], r->tmpvec + m * H, r->emod + m * 3 * H, 1));
        TRY(linear_op(r, &r->single[m], r->tmpvec + m * H, r->smod + m * 3 * H, 1));
        HIP_CHECK(hipMemcpy(r->in, inputs[m], (size_t)count[m] * r->input[m].cols * 4,
                            hipMemcpyHostToDevice));
        TRY(linear_op(r, &r->input[m], r->in, r->x + (size_t)start[m] * H, count[m]));
        HIP_CHECK(hipMemcpy(r->ids + start[m] * 4, allids[m], (size_t)count[m] * 16,
                            hipMemcpyHostToDevice));
    }
    TRY(linear_op(r, &r->final_mod, r->tmpvec, r->fmod, 1));
    for (int d = 0; d < 5; d++)
        for (int m = 0; m < 3; m++)
            if (count[m])
                TRY(block_op(r, &r->early_block[m][d], r->x + (size_t)start[m] * H,
                             r->ids + start[m] * 4, r->mode + start[m], r->emod, count[m]));
    for (int d = 0; d < 28; d++) {
        TRY(block_op(r, &r->single_block[d], r->x, r->ids, r->mode, r->smod, n));
        if (r->verbose > 1)
            fprintf(stderr, "flux3: shared block %d/28\n", d + 1);
    }
    TRY(ln_op(r, r->ln, r->x, r->fmod, r->mode, ni));
    TRY(linear_op(r, &r->final_out, r->ln, r->ap, ni));
    HIP_CHECK(hipMemcpy(out, r->ap, (size_t)ni * 128 * 4, hipMemcpyDeviceToHost));
    for (size_t j = 0; j < (size_t)ni * 128; j++)
        if (!isfinite(out[j])) {
            fprintf(stderr, "flux3: nonfinite velocity at %zu\n", j);
            return -1;
        }
    return 0;
}
int hip_flux3_selftest(hip_flux3_runner *r) {
    if (!r || r->loaded || r->nowned)
        return -1;
    // Non-square/tail GEMM tests against a scalar CPU dot product; catches
    // WMMA layout, E4M3 exponent 15/subnormal decoding and row-scale mistakes.
    enum { M = 7, N = 19, K = 48 };
    float x[M * K], got[M * N], sc[N];
    uint16_t wb[N * K];
    unsigned char w8[N * K];
    for (int i = 0; i < M * K; i++)
        x[i] = bf_host(sinf(i * .17f) * .8f);
    for (int j = 0; j < N; j++) {
        sc[j] = ldexpf(1.f, (j % 7) - 5);
        for (int k = 0; k < K; k++) {
            float v = bf_host(cosf((j * K + k) * .11f));
            uint32_t u;
            memcpy(&u, &v, 4);
            wb[j * K + k] = u >> 16;
            unsigned z = (j * K + k) % 127;
            w8[j * K + k] = (unsigned char)(z | ((k & 1) ? 128 : 0));
        }
    }
    TRY(workspace(r, M));
    HIP_CHECK(hipMemcpy(r->in, x, sizeof(x), hipMemcpyHostToDevice));
    weight q = {0};
    q.rows = N;
    q.cols = K;
    q.data = upload(r, wb, sizeof(wb));
    if (!q.data)
        return -1;
    TRY(linear_op(r, &q, r->in, r->ap, M));
    HIP_CHECK(hipMemcpy(got, r->ap, sizeof(got), hipMemcpyDeviceToHost));
    float worst = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                uint32_t u = (uint32_t)wb[j * K + k] << 16;
                float v;
                memcpy(&v, &u, 4);
                sum += x[i * K + k] * v;
            }
            float e = fabsf(got[i * N + j] - bf_host(sum));
            if (e > worst)
                worst = e;
        }
    fprintf(stderr, "flux3: BF16 GEMM selftest max error %.6g\n", worst);
    if (worst > 0.02f)
        return -1;
    q.fp8 = 1;
    q.data = upload(r, w8, sizeof(w8));
    q.scale = upload(r, sc, sizeof(sc));
    if (!q.data || !q.scale)
        return -1;
    TRY(linear_op(r, &q, r->in, r->ap, M));
    HIP_CHECK(hipMemcpy(got, r->ap, sizeof(got), hipMemcpyDeviceToHost));
    worst = 0;
    for (int i = 0; i < M; i++) {
        float a = 0;
        for (int k = 0; k < K; k++)
            a = fmaxf(a, fabsf(x[i * K + k]));
        a = fmaxf(a / 448, 1e-12f);
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                float v = x[i * K + k] / a, av = fminf(fabsf(v), 448);
                int e;
                frexpf(av, &e);
                float step = ldexpf(1.f, e - 4 > -9 ? e - 4 : -9);
                v = copysignf(fminf(nearbyintf(av / step) * step, 448), v);
                unsigned z = w8[j * K + k], ex = (z >> 3) & 15, ma = z & 7;
                float wf = ex ? ldexpf(1 + ma / 8.f, (int)ex - 7) : ma / 512.f;
                if (z & 128)
                    wf = -wf;
                sum += v * wf;
            }
            float want = bf_host(sum * a * sc[j]);
            float e = fabsf(got[i * N + j] - want) / fmaxf(1, fabsf(want));
            if (e > worst)
                worst = e;
        }
    }
    fprintf(stderr, "flux3: row-scaled FP8 GEMM selftest max relative error %.6g\n", worst);
    if (worst >= .01f)
        return -1;
    // Uniform attention has an exact, independent mean-value oracle.
    float values[2 * H], answer[2 * H];
    for (int j = 0; j < 2 * H; j++)
        values[j] = j < H ? .5f : 1.5f;
    HIP_CHECK(hipMemset(r->q, 0, 2 * H * 4));
    HIP_CHECK(hipMemset(r->k, 0, 2 * H * 4));
    HIP_CHECK(hipMemcpy(r->v, values, sizeof(values), hipMemcpyHostToDevice));
    int na = 2;
    void *aa[] = {&r->att, &r->q, &r->k, &r->v, &na};
    TRY(launch(r->attention, 24, 1, 128, aa));
    HIP_CHECK(hipMemcpy(answer, r->att, sizeof(answer), hipMemcpyDeviceToHost));
    for (int j = 0; j < 2 * H; j++)
        if (answer[j] != 1.f)
            return -1;
    // Unit input RMS-normalizes to approximately one. All four distinct IDs
    // must rotate adjacent pairs independently, including negative axis four.
    float unit[H], scale[128], positions[4] = {.25f, .5f, 1.f, -2.f};
    for (int j = 0; j < H; j++)
        unit[j] = 1.f;
    for (int j = 0; j < 128; j++)
        scale[j] = 1.f;
    float *dscale = upload(r, scale, sizeof(scale));
    if (!dscale)
        return -1;
    HIP_CHECK(hipMemcpy(r->q, unit, sizeof(unit), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(r->ids, positions, sizeof(positions), hipMemcpyHostToDevice));
    na = 1;
    void *ra[] = {&r->q, &dscale, &r->ids, &na};
    TRY(launch(r->normrope, 24, 1, 128, ra));
    HIP_CHECK(hipMemcpy(unit, r->q, sizeof(unit), hipMemcpyDeviceToHost));
    worst = 0;
    for (int j = 0; j < H; j++) {
        int d = j % 128, p = d & ~1;
        double a = positions[d / 32] * pow(10000., -(double)(p % 32) / 32);
        float want = bf_host((float)((cos(a) + ((d & 1) ? 1 : -1) * sin(a)) / sqrt(1.000001)));
        worst = fmaxf(worst, fabsf(unit[j] - want));
    }
    fprintf(stderr, "flux3: attention mean and four-axis RoPE selftests, max error %.6g\n", worst);
    if (worst > .016f)
        return -1;
    // Selftest uploads must not prevent a subsequent checkpoint load.
    for (int i = 0; i < r->nowned; i++)
        hipFree(r->owned[i]);
    r->nowned = 0;
    return 0;
}
void hip_flux3_free(hip_flux3_runner *r) {
    if (!r)
        return;
    hipSetDevice(r->device);
    if (r->workspace)
        hipFree(r->workspace);
    for (int i = 0; i < r->nowned; i++)
        hipFree(r->owned[i]);
    if (r->module)
        hipModuleUnload(r->module);
    free(r);
}
