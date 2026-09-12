/* Internal HIP stream ownership for Qwen4 prefill. Include after rocew.h. */
#ifndef QWEN4_MOE_STAGE_H
#define QWEN4_MOE_STAGE_H

typedef struct {
    hipEvent_t ready, done;
    int uploaded, consumed;
} qwen4_moe_fence;

static int qwen4_moe_fence_init(qwen4_moe_fence *f) {
    if (hipEventCreateWithFlags(&f->ready, hipEventDisableTiming) != hipSuccess)
        return -1;
    if (hipEventCreateWithFlags(&f->done, hipEventDisableTiming) != hipSuccess) {
        hipEventDestroy(f->ready);
        f->ready = NULL;
        return -1;
    }
    return 0;
}

/* Host metadata may be rewritten only after its previous DMA completes.
 * Device storage may be overwritten only after its last consumer completes.
 * Waiting on ready on the host still permits overlap with that consumer. */
static int qwen4_moe_acquire(qwen4_moe_fence *f, hipStream_t copy) {
    if (f->uploaded && hipEventSynchronize(f->ready) != hipSuccess) return -1;
    if (f->consumed && hipStreamWaitEvent(copy, f->done, 0) != hipSuccess) return -1;
    return 0;
}

static int qwen4_moe_publish(qwen4_moe_fence *f, hipStream_t copy,
                             hipStream_t compute) {
    if (hipEventRecord(f->ready, copy) != hipSuccess) return -1;
    f->uploaded = 1;
    return hipStreamWaitEvent(compute, f->ready, 0) == hipSuccess ? 0 : -1;
}

static int qwen4_moe_consumed(qwen4_moe_fence *f, hipStream_t compute) {
    if (hipEventRecord(f->done, compute) != hipSuccess) return -1;
    f->consumed = 1;
    return 0;
}

/* Call only after BOTH streams have been drained, including error paths. */
static void qwen4_moe_fence_reset(qwen4_moe_fence *f) {
    f->uploaded = f->consumed = 0;
}

static void qwen4_moe_fence_free(qwen4_moe_fence *f) {
    if (f->ready) hipEventDestroy(f->ready);
    if (f->done) hipEventDestroy(f->done);
    memset(f, 0, sizeof(*f));
}

typedef struct {
    void *gate, *up, *down, *q8;
    /* [expert map, task expert IDs, task positions], independent of router
     * scratch. Host storage is pinned even for pageable model weights. */
    int *host, *device;
    qwen4_moe_fence fence;
} qwen4_moe_bank;

static int qwen4_moe_bank_init(qwen4_moe_bank *b, int slots,
        size_t sg, size_t su, size_t sd, size_t q8_bytes, size_t metadata_bytes) {
    if (qwen4_moe_fence_init(&b->fence) ||
        hipMalloc(&b->gate, (size_t)slots * sg) != hipSuccess ||
        hipMalloc(&b->up, (size_t)slots * su) != hipSuccess ||
        hipMalloc(&b->down, (size_t)slots * sd) != hipSuccess ||
        (q8_bytes && hipMalloc(&b->q8, q8_bytes) != hipSuccess) ||
        hipMalloc((void **)&b->device, metadata_bytes) != hipSuccess ||
        hipHostMalloc((void **)&b->host, metadata_bytes, hipHostMallocDefault) != hipSuccess)
        return -1;
    return 0;
}

/* Also accepts partially initialized banks; caller has drained work first. */
static void qwen4_moe_bank_free(qwen4_moe_bank *b) {
    if (b->gate) hipFree(b->gate);
    if (b->up) hipFree(b->up);
    if (b->down) hipFree(b->down);
    if (b->q8) hipFree(b->q8);
    if (b->device) hipFree(b->device);
    if (b->host) hipHostFree(b->host);
    qwen4_moe_fence_free(&b->fence);
    memset(b, 0, sizeof(*b));
}
#endif
