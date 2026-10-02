/* Separate denoising progress from loader/tiling callbacks, retaining SDK fallback. */
#include "progress.h"
#include "core/util.h"
static hv15_sample_progress_callback sample_callback=nullptr;
static void *sample_user=nullptr;
extern "C" SD_API void hv15_set_sample_progress_callback(hv15_sample_progress_callback callback, void *user) {
    sample_callback=callback;
    sample_user=user;
}
extern "C" SD_API void hv15_report_sample_progress(int step, int total, float seconds) {
    if (sample_callback) sample_callback(step,total,seconds,sample_user);
    else pretty_progress(step,total,seconds);
}
