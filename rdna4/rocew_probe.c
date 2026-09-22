#include "rocew.h"
#include <stdio.h>

int main(void) {
    int rc = rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC);
    printf("rocew_init=%d hip=%d hiprtc=%d\n", rc,
           rocewHipAvailable(), rocewHiprtcAvailable());
    if (rc != ROCEW_SUCCESS || !hipGetDeviceCount) return 1;
    int count = -1;
    hipError_t e = hipGetDeviceCount(&count);
    printf("device_count_rc=%d count=%d\n", e, count);
    for (int i = 0; e == hipSuccess && i < count; ++i) {
        hipDeviceProp_t p;
        e = hipGetDeviceProperties(&p, i);
        printf("device=%d props_rc=%d name=%s arch=%s\n", i, e,
               e == hipSuccess ? p.name : "<error>",
               e == hipSuccess ? p.gcnArchName : "<error>");
    }
    if (rocewHiprtcAvailable() && hiprtcVersion) {
        int major = 0, minor = 0;
        int vrc = hiprtcVersion(&major, &minor);
        printf("hiprtc_version_rc=%d version=%d.%d\n", vrc, major, minor);
    }
    return e == hipSuccess ? 0 : 2;
}
