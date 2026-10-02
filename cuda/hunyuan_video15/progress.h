#ifndef HV15_PROGRESS_H
#define HV15_PROGRESS_H
#include "stable-diffusion.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef void (*hv15_sample_progress_callback)(int, int, float, void *);
SD_API void hv15_set_sample_progress_callback(hv15_sample_progress_callback callback, void *user);
SD_API void hv15_report_sample_progress(int step, int total, float seconds);
#ifdef __cplusplus
}
#endif
#endif
