/* CPU-only fault injection for the production CUDA/HIP host implementation.
 * Never initialize cuew or a device. Driver entry points are CPU doubles. */
#include <assert.h>
#include "vhuman_deformer_cuda.c"

static int allocations, fail_alloc, fail_sync, fail_download, fail_event, fail_record, fail_upload, fail_launch;
static int stream_waits, live_events;
static CUresult CUDAAPI alloc_device(CUdeviceptr *p, size_t n) {
    if (++allocations == fail_alloc) return CUDA_ERROR_OUT_OF_MEMORY;
    *p = (CUdeviceptr)(uintptr_t)malloc(n);
    return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
static CUresult CUDAAPI free_device(CUdeviceptr p) { free((void *)(uintptr_t)p); return CUDA_SUCCESS; }
static CUresult CUDAAPI alloc_host(void **p, size_t n) {
    if (++allocations == fail_alloc) return CUDA_ERROR_OUT_OF_MEMORY;
    *p = malloc(n);
    return *p ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
static CUresult CUDAAPI free_host(void *p) { free(p); return CUDA_SUCCESS; }
static CUresult CUDAAPI context(CUcontext ctx) { (void)ctx; return CUDA_SUCCESS; }
static CUresult CUDAAPI stream_wait(CUstream s) { (void)s; ++stream_waits; return CUDA_SUCCESS; }
static CUresult CUDAAPI event_create(CUevent *e, unsigned flags) {
    (void)flags;
    if (fail_event) return CUDA_ERROR_OUT_OF_MEMORY;
    *e = (CUevent)malloc(1); ++live_events; return CUDA_SUCCESS;
}
static CUresult CUDAAPI event_destroy(CUevent e) { free(e); --live_events; return CUDA_SUCCESS; }
static CUresult CUDAAPI event_record(CUevent e, CUstream s) {
    (void)e; (void)s; return fail_record ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
static CUresult CUDAAPI event_wait(CUevent e) { (void)e; return fail_sync ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS; }
static CUresult CUDAAPI elapsed(float *ms, CUevent a, CUevent b) { (void)a; (void)b; *ms = 1; return CUDA_SUCCESS; }
static CUresult CUDAAPI upload(CUdeviceptr p, const void *src, size_t n) {
    if (fail_upload) return CUDA_ERROR_INVALID_VALUE;
    memcpy((void *)(uintptr_t)p, src, n); return CUDA_SUCCESS;
}
static CUresult CUDAAPI upload_async(CUdeviceptr p, const void *src, size_t n, CUstream s) {
    (void)s; return upload(p, src, n);
}
static CUresult CUDAAPI download(void *dst, CUdeviceptr src, size_t n) {
    if (fail_download) return CUDA_ERROR_INVALID_VALUE;
    memcpy(dst, (void *)(uintptr_t)src, n); return CUDA_SUCCESS;
}
static CUresult CUDAAPI launch(CUfunction fn, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                              unsigned by, unsigned bz, unsigned sm, CUstream s, void **args, void **extra) {
    (void)fn; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)sm;
    (void)s; (void)args; (void)extra;
    return fail_launch ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
static vh_gpu *make_gpu(vh_deformer *d) {
    vh_gpu *g = calloc(1, sizeof(*g));
    g->d = d; g->M = 1; g->J = 1; g->V = 3; g->C = 1; g->nc = 3;
    assert(event_create(&g->upload_ready, 0) == CUDA_SUCCESS);
    return g;
}

int main(int argc, char **argv) {
    assert(argc == 2 || argc == 3);
    if (argc == 3) {
        vh_deformer *bad = vh_deformer_load(argv[1]);
        assert(!bad);
        return 0;
    }
    cuMemAlloc = alloc_device; cuMemFree = free_device;
    cuMemAllocHost = alloc_host; cuMemFreeHost = free_host;
    cuCtxSetCurrent = context; cuStreamSynchronize = stream_wait;
    cuEventCreate = event_create; cuEventDestroy = event_destroy;
    cuEventRecord = event_record; cuEventSynchronize = event_wait; cuEventElapsedTime = elapsed;
    cuMemcpyHtoD = upload; cuMemcpyHtoDAsync = upload_async; cuMemcpyDtoH = download;
    cuLaunchKernel = launch;
    vh_deformer *d = vh_deformer_load(argv[1]); assert(d);

    for (int nth = 1; nth <= 6; ++nth) {
        vh_gpu *g = make_gpu(d);
        assert(!ensure(g, 1));
        fail_alloc = allocations + nth;
        assert(ensure(g, 65) < 0);
        assert(!g->cap_frames && !g->w && !g->skin && !g->out && !g->h_w && !g->h_skin);
        fail_alloc = 0;
        assert(!ensure(g, 1) && g->cap_frames == 64 && g->out);
        vh_gpu_free(g);
    }

    vh_gpu *g = make_gpu(d);
    float controls[1] = {0}, out[9];
    assert(!vh_gpu_eval_batch(g, controls, 1, 0, out, NULL));
    int *faults[] = {&fail_sync, &fail_download, &fail_event, &fail_record, &fail_upload, &fail_launch};
    for (size_t i = 0; i < sizeof(faults)/sizeof(faults[0]); ++i) {
        if (faults[i] == &fail_sync) g->upload_pending = 0; /* fail completion, after launch */
        *faults[i] = 1;
        int before = stream_waits;
        assert(vh_gpu_eval_batch(g, controls, 1, 0, out, NULL) < 0);
        if (faults[i] == &fail_upload) assert(stream_waits > before);
        *faults[i] = 0;
        assert(live_events == 1); /* only the persistent upload event */
        assert(!vh_gpu_eval_batch(g, controls, 1, 0, out, NULL));
    }
    /* Exact-size EOF CSR input under ASan: copying one extra int fails. */
    int *neighbors = malloc(2 * sizeof(int)); neighbors[0] = 1; neighbors[1] = 0;
    CUdeviceptr p = upload_checked(neighbors, 2 * sizeof(int)); assert(p); CU_FREE(p); free(neighbors);
    p = upload_checked(NULL, 0); assert(p); CU_FREE(p);
    fail_upload = 1; p = upload_checked(controls, sizeof(controls)); assert(!p); fail_upload = 0;
    vh_gpu_free(g); vh_deformer_free(d); assert(!live_events);
    puts("CPU driver fault injection passed (no device initialization)");
    return 0;
}
