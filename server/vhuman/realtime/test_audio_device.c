/* Native callback test without opening a device; checks exact sample accounting. */
#include "src/audio/pcm_ring.c"
#include "src/audio/device.c"
#include <assert.h>
#include <stdio.h>
int main(void) {
    vh_pcm_ring *ring = vh_pcm_create(48000);
    vh_audio *audio = calloc(1, sizeof(*audio));
    assert(ring && audio);
    audio->ring = ring;
    atomic_init(&audio->write_at, 0); atomic_init(&audio->read_at, 0);
    atomic_init(&audio->submitted, 0); atomic_init(&audio->overflow, 0); atomic_init(&audio->origin_ns, -1);
    float input[240], output[480];
    for (int i = 0; i < 240; ++i) input[i] = (float)i;
    assert(vh_pcm_write(ring, input, 240));
    PaStreamCallbackTimeInfo time = {0, 0, 1};
    callback(NULL, output, 480, &time, 0, audio);
    for (int i = 0; i < 240; ++i) assert(output[i] == input[i] && output[i+240] == 0);
    vh_audio_event event;
    assert(vh_audio_event_pop(audio, &event) == 1 && event.count == 240 && !event.silence && event.speech_start == 0);
    assert(vh_audio_event_pop(audio, &event) == 1 && event.count == 240 && event.silence && event.speech_start == 240);
    assert(vh_pcm_write(ring, input, 240));
    time.outputBufferDacTime = 1.04; /* 20ms host gap after the prior20ms block */
    callback(NULL, output, 480, &time, paOutputUnderflow, audio);
    assert(vh_audio_event_pop(audio, &event) == 1 && event.count == 480 && event.silence && event.speech_start == 240);
    assert(vh_audio_event_pop(audio, &event) == 1 && event.count == 240 && !event.silence && event.speech_start == 240);
    assert(atomic_load(&audio->submitted) == 1440 && audio->speech == 480);
    free(audio); vh_pcm_free(ring);
    puts("native callback: queue underrun and host DAC gap accounting OK");
    return 0;
}
