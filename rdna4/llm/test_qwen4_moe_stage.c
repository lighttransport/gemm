/* CPU-only adversarial scheduler for the production bank ownership helpers.
 * Copies retain host pointers, kernels run late, and waits capture event
 * generations. Eager memcpy mocks would hide the bugs this test targets. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
enum { hipSuccess = 0, hipEventDisableTiming = 1, hipHostMallocDefault = 0 };
typedef struct stream *hipStream_t;
typedef struct event { hipStream_t stream; int end; } *hipEvent_t;
typedef struct {
    int kind, end, expected;
    hipStream_t dependency;
    void *dst;
    const void *src;
    size_t bytes;
    int *map;
} operation;
struct stream { operation ops[4096]; int count, cursor; };
static int fail_at, calls, live_allocations, live_events, checks;
static int api_fail(void) { return ++calls == fail_at; }
static int hipEventCreateWithFlags(hipEvent_t *e, int flags) {
    assert(flags == hipEventDisableTiming);
    if (api_fail()) return 1;
    *e = calloc(1, sizeof(**e)); assert(*e); ++live_events; return 0;
}
static int hipEventDestroy(hipEvent_t e) { free(e); --live_events; return 0; }
static void run(hipStream_t s, int end) {
    while (s->cursor < end) {
        operation o = s->ops[s->cursor++];
        if (o.kind == 0) memcpy(o.dst, o.src, o.bytes);
        else if (o.kind == 1) run(o.dependency, o.end);
        else if (o.kind == 2) {
            /* Model compact->padded Q8_0: insert two zero padding bytes. */
            unsigned char *d = o.dst;
            const unsigned char *src = o.src;
            memcpy(d, src, 2); d[2] = d[3] = 0;
            memcpy(d + 4, src + 2, 32);
        } else {
            const unsigned char *d = o.src;
            assert(o.map[0] == o.expected);
            assert(d[0] == o.expected && d[1] == o.expected);
            assert(d[2] == 0 && d[3] == 0);
            for (int i = 4; i < 36; ++i) assert(d[i] == o.expected);
            ++checks;
        }
    }
}
static int hipEventRecord(hipEvent_t e, hipStream_t s) {
    if (api_fail()) return 1;
    e->stream = s; e->end = s->count; return 0;
}
static int hipEventSynchronize(hipEvent_t e) {
    if (api_fail()) return 1;
    assert(e->stream); run(e->stream, e->end); return 0;
}
static int hipStreamWaitEvent(hipStream_t s, hipEvent_t e, int flags) {
    assert(flags == 0 && e->stream);
    if (api_fail()) return 1;
    s->ops[s->count++] = (operation){.kind=1, .dependency=e->stream, .end=e->end};
    return 0;
}
static int hipMalloc(void **p, size_t bytes) {
    if (api_fail()) return 1;
    *p = malloc(bytes); assert(*p); ++live_allocations; return 0;
}
static int hipFree(void *p) { free(p); --live_allocations; return 0; }
static int hipHostMalloc(void **p, size_t bytes, int flags) {
    assert(flags == hipHostMallocDefault); return hipMalloc(p, bytes);
}
static int hipHostFree(void *p) { return hipFree(p); }
#include "qwen4_moe_stage.h"

static void copy(hipStream_t s, void *dst, const void *src, size_t n) {
    s->ops[s->count++] = (operation){.dst=dst, .src=src, .bytes=n};
}
static void exercise(int overlap) {
    struct stream upload = {0}, compute = {0};
    qwen4_moe_bank banks[2] = {0};
    unsigned char weights[40][34];
    for (int i = 0; i < 2; ++i)
        assert(qwen4_moe_bank_init(&banks[i], 1, 36, 36, 36, 34, sizeof(int)) == 0);
    for (int wave = 1; wave <= 40; ++wave) {
        qwen4_moe_bank *b = &banks[overlap ? wave % 2 : 0];
        assert(qwen4_moe_acquire(&b->fence, &upload) == 0);
        b->host[0] = wave;
        memset(weights[wave-1], wave, 34);
        copy(&upload, b->q8, weights[wave-1], 34);
        upload.ops[upload.count++] = (operation){.kind=2, .dst=b->down, .src=b->q8};
        copy(&upload, b->device, b->host, sizeof(int));
        assert(qwen4_moe_publish(&b->fence, &upload, &compute) == 0);
        compute.ops[compute.count++] = (operation){.kind=3, .src=b->down,
                                                  .map=b->device, .expected=wave};
        assert(qwen4_moe_consumed(&b->fence, &compute) == 0);
        /* Alternate copy-first and compute-first scheduling at request resets. */
        if (wave % 10 == 0) {
            if (wave % 20) run(&upload, upload.count);
            run(&compute, compute.count); run(&upload, upload.count);
            for (int i = 0; i < 2; ++i) qwen4_moe_fence_reset(&banks[i].fence);
        }
    }
    run(&upload, upload.count); run(&compute, compute.count);
    for (int i = 0; i < 2; ++i) qwen4_moe_bank_free(&banks[i]);
    assert(live_allocations == 0 && live_events == 0);
}
static void failure_tests(void) {
    /* Every allocation/event creation may fail; partially built banks clean up. */
    for (int at = 1; at <= 8; ++at) {
        qwen4_moe_bank b = {0}; calls = 0; fail_at = at;
        assert(qwen4_moe_bank_init(&b, 1, 36, 36, 36, 34, 4) == -1);
        qwen4_moe_bank_free(&b);
        assert(!live_allocations && !live_events);
    }
    for (int phase = 0; phase < 3; ++phase) for (int at = 1; at <= (phase == 2 ? 1 : 2); ++at) {
        qwen4_moe_fence f = {0}; struct stream upload = {0}, compute = {0};
        calls = 0; fail_at = 0;
        assert(qwen4_moe_fence_init(&f) == 0);
        assert(qwen4_moe_publish(&f, &upload, &compute) == 0);
        assert(qwen4_moe_consumed(&f, &compute) == 0);
        calls = 0; fail_at = at;
        int rc = phase == 0 ? qwen4_moe_acquire(&f, &upload) :
                 phase == 1 ? qwen4_moe_publish(&f, &upload, &compute) :
                              qwen4_moe_consumed(&f, &compute);
        assert(rc == -1);
        fail_at = 0;
        run(&upload, upload.count); run(&compute, compute.count);
        qwen4_moe_fence_reset(&f); qwen4_moe_fence_free(&f);
        assert(!live_events);
    }
}
int main(void) {
    exercise(0); exercise(1); failure_tests();
    assert(checks == 80);
    puts("Qwen4 bank ownership: 80 delayed consumers, repack, reset, failures: PASS");
    return 0;
}
