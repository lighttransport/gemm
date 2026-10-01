/* Single producer/single consumer PCM queue. Reset/free only with both stopped.
 * Callback read: no allocation, locks, logging, Python, or model execution. */
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct vh_pcm_ring {
    float *pcm;
    uint64_t capacity;
    _Atomic uint64_t write_at, read_at;
} vh_pcm_ring;

vh_pcm_ring *vh_pcm_create(uint64_t capacity) {
    if (!capacity || capacity > SIZE_MAX / sizeof(float)) return NULL;
    vh_pcm_ring *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->pcm = calloc((size_t)capacity, sizeof(float));
    if (!r->pcm) { free(r); return NULL; }
    r->capacity = capacity;
    atomic_init(&r->write_at, 0);
    atomic_init(&r->read_at, 0);
    if (!atomic_is_lock_free(&r->write_at) || !atomic_is_lock_free(&r->read_at)) {
        free(r->pcm); free(r); return NULL;
    }
    return r;
}
void vh_pcm_free(vh_pcm_ring *r) { if (r) { free(r->pcm); free(r); } }
void vh_pcm_reset(vh_pcm_ring *r) {
    atomic_store_explicit(&r->write_at, 0, memory_order_relaxed);
    atomic_store_explicit(&r->read_at, 0, memory_order_relaxed);
}
/* Producer-side depth; the consumer can only make more space. */
uint64_t vh_pcm_depth(vh_pcm_ring *r) {
    uint64_t w = atomic_load_explicit(&r->write_at, memory_order_relaxed);
    uint64_t q = atomic_load_explicit(&r->read_at, memory_order_acquire);
    return w - q;
}
int vh_pcm_write(vh_pcm_ring *r, const float *pcm, uint64_t count) {
    uint64_t w = atomic_load_explicit(&r->write_at, memory_order_relaxed);
    uint64_t q = atomic_load_explicit(&r->read_at, memory_order_acquire);
    if (count > r->capacity - (w - q) || count > UINT64_MAX - w) return 0;
    for (uint64_t i = 0; i < count; ++i) r->pcm[(w + i) % r->capacity] = pcm[i];
    atomic_store_explicit(&r->write_at, w + count, memory_order_release);
    return 1;
}
uint64_t vh_pcm_read(vh_pcm_ring *r, float *out, uint64_t count) {
    uint64_t q = atomic_load_explicit(&r->read_at, memory_order_relaxed);
    uint64_t w = atomic_load_explicit(&r->write_at, memory_order_acquire);
    uint64_t n = w - q < count ? w - q : count;
    for (uint64_t i = 0; i < n; ++i) out[i] = r->pcm[(q + i) % r->capacity];
    memset(out + n, 0, (size_t)(count - n) * sizeof(float));
    atomic_store_explicit(&r->read_at, q + n, memory_order_release);
    return n;
}
