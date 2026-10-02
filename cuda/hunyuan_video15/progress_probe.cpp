/* Check channel separation using the pinned SDK's actual loader callback path. */
#include <cstdio>
#include "progress.h"
#include "core/util.h"
static int component_calls=0, sample_calls=0;
int main() {
    sd_set_progress_callback([](int,int,float,void *) { ++component_calls; },nullptr);
    hv15_set_sample_progress_callback([](int,int,float,void *) { ++sample_calls; },nullptr);
    pretty_bytes_progress(12,12,4096,.001f);
    if (component_calls!=1 || sample_calls!=0) return 1;
    hv15_report_sample_progress(1,12,2.f);
    if (component_calls!=1 || sample_calls!=1) return 1;
    hv15_set_sample_progress_callback(nullptr,nullptr);
    hv15_report_sample_progress(2,12,2.f);
    if (component_calls!=2 || sample_calls!=1) return 1;
    sd_set_progress_callback(nullptr,nullptr);
    std::puts("loader/sample progress channels and SDK fallback: PASS");
    return 0;
}
