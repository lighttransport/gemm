#include "rocew.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--json") && !strcmp(argv[2], "--device")) {
        char *end;
        long device = strtol(argv[3], &end, 10);
        if (*end || device < 0 || device > 65535) return 2;
        if (rocewInit(ROCEW_INIT_HIP) != ROCEW_SUCCESS) return 1;
        hipDeviceProp_t p;
        size_t free_bytes, total_bytes;
        if (hipSetDevice((int)device) || hipGetDeviceProperties(&p, (int)device) ||
            hipMemGetInfo(&free_bytes, &total_bytes)) return 1;
        /* Escape the device name, which is supplied by the runtime. */
        printf("{\"backend\":\"rocm\",\"device\":%ld,\"name\":\"", device);
        for (const unsigned char *s = (const unsigned char *)p.name; *s; s++) {
            if (*s < 32) printf("\\u%04x", *s);
            else { if (*s == '"' || *s == '\\') putchar('\\'); putchar(*s); }
        }
        printf("\",\"arch\":\"%s\",\"free_mib\":%zu,\"total_mib\":%zu}\n",
               p.gcnArchName, free_bytes >> 20, total_bytes >> 20);
        return 0;
    }
    if (argc != 1) return 2;
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
