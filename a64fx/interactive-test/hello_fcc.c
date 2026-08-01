#include <stdio.h>
#include <stdint.h>

int main(void) {
    uint64_t acc = 0;
    for (uint64_t i = 1; i <= 1000000; ++i) {
        acc += (i * 2654435761u) ^ (i >> 3);
    }
    printf("hello from fcc native A64FX test\n");
    printf("acc=%llu\n", (unsigned long long)acc);
    printf("SENTINEL interactive_fcc_native=OK\n");
    return 0;
}
