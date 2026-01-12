#include <stdio.h>
extern const char* ptx_gemm_tcgen05_s8;

// Declare it from int8_gemm.c
static const char* ptx_gemm_tcgen05_s8 =
#include "tcgen05_ptx.h"
;

int main() {
    int line = 1;
    const char* p = ptx_gemm_tcgen05_s8;
    while (*p) {
        printf("%3d: ", line);
        while (*p && *p != '\n') {
            putchar(*p++);
        }
        putchar('\n');
        if (*p == '\n') { p++; line++; }
    }
    return 0;
}
