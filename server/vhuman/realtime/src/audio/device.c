/* Native PortAudio sink. Metadata ring is SPSC; callback never calls Python. */
#include <portaudio.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
typedef struct vh_pcm_ring vh_pcm_ring;
uint64_t vh_pcm_read(vh_pcm_ring *, float *, uint64_t);
#define EVENT_CAPACITY 4096
typedef struct { uint64_t count, speech_start; int silence; } vh_audio_event;
typedef struct {
    PaStream *stream;
    vh_pcm_ring *ring;
    vh_audio_event events[EVENT_CAPACITY];
    _Atomic uint64_t write_at, read_at, submitted;
    _Atomic int overflow;
    _Atomic int64_t origin_ns;
    uint64_t speech;
} vh_audio;
static void emit(vh_audio *a, uint64_t count, int silence) {
    if (!count) return;
    uint64_t w = atomic_load_explicit(&a->write_at, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&a->read_at, memory_order_acquire);
    if (w - r == EVENT_CAPACITY) {
        atomic_store_explicit(&a->overflow, 1, memory_order_release);
        return;
    }
    a->events[w % EVENT_CAPACITY] = (vh_audio_event){ count, a->speech, silence };
    atomic_store_explicit(&a->write_at, w + 1, memory_order_release);
}
static int callback(const void *input, void *output, unsigned long frames,
                    const PaStreamCallbackTimeInfo *info, PaStreamCallbackFlags flags, void *user) {
    (void)input; (void)flags;
    vh_audio *a = user;
    if (atomic_load_explicit(&a->origin_ns, memory_order_relaxed) < 0)
        atomic_store_explicit(&a->origin_ns, (int64_t)(info->outputBufferDacTime * 1e9), memory_order_release);
    int64_t origin = atomic_load_explicit(&a->origin_ns, memory_order_relaxed);
    uint64_t submitted = atomic_load_explicit(&a->submitted, memory_order_relaxed);
    int64_t expected_ns = origin + (int64_t)((submitted / 24000) * 1000000000 +
                                            (submitted % 24000) * 1000000000 / 24000);
    int64_t gap_ns = (int64_t)(info->outputBufferDacTime * 1e9) - expected_ns;
    /* A host underrun can delay the actual DAC timestamp beyond our submitted
     * samples. Represent that physical gap as silence, too. Ignore sub-block
     * timestamp jitter unless the host explicitly reports an underrun. */
    if (gap_ns > 0 && ((flags & paOutputUnderflow) || gap_ns >= 20000000)) {
        uint64_t gap = (uint64_t)gap_ns * 24000 / 1000000000;
        emit(a, gap, 1);
        atomic_fetch_add_explicit(&a->submitted, gap, memory_order_release);
    }
    uint64_t n = vh_pcm_read(a->ring, output, frames);
    emit(a, n, 0);
    a->speech += n;
    emit(a, frames - n, 1);
    atomic_fetch_add_explicit(&a->submitted, frames, memory_order_release);
    return paContinue;
}
vh_audio *vh_audio_open(vh_pcm_ring *ring, int *error) {
    *error = Pa_Initialize();
    if (*error != paNoError) return NULL;
    vh_audio *a = calloc(1, sizeof(*a));
    if (!a) { Pa_Terminate(); *error = paInsufficientMemory; return NULL; }
    a->ring = ring;
    atomic_init(&a->write_at, 0); atomic_init(&a->read_at, 0);
    atomic_init(&a->submitted, 0); atomic_init(&a->origin_ns, -1); atomic_init(&a->overflow, 0);
    *error = Pa_OpenDefaultStream(&a->stream, 0, 1, paFloat32, 24000, 480, callback, a);
    if (*error != paNoError) { free(a); Pa_Terminate(); return NULL; }
    return a;
}
int vh_audio_start(vh_audio *a) { return Pa_StartStream(a->stream); }
int vh_audio_event_pop(vh_audio *a, vh_audio_event *out) {
    if (atomic_load_explicit(&a->overflow, memory_order_acquire)) return -1;
    uint64_t r = atomic_load_explicit(&a->read_at, memory_order_relaxed);
    uint64_t w = atomic_load_explicit(&a->write_at, memory_order_acquire);
    if (r == w) return 0;
    *out = a->events[r % EVENT_CAPACITY];
    atomic_store_explicit(&a->read_at, r + 1, memory_order_release);
    return 1;
}
uint64_t vh_audio_played(vh_audio *a) {
    int64_t origin = atomic_load_explicit(&a->origin_ns, memory_order_acquire);
    int64_t now = (int64_t)(Pa_GetStreamTime(a->stream) * 1e9);
    uint64_t delta = origin < 0 || now < origin ? 0 : (uint64_t)(now - origin);
    uint64_t played = delta / 1000000000 * 24000 + delta % 1000000000 * 24000 / 1000000000;
    uint64_t submitted = atomic_load_explicit(&a->submitted, memory_order_acquire);
    return played < submitted ? played : submitted;
}
void vh_audio_close(vh_audio *a) {
    if (!a) return;
    Pa_AbortStream(a->stream); Pa_CloseStream(a->stream);
    free(a); Pa_Terminate();
}
