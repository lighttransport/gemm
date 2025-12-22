#include <stdio.h>
#include <stdlib.h>

int main() {
    size_t count = 16384;
    float* input = aligned_alloc(64, count * sizeof(float));
    float* output = aligned_alloc(64, count * sizeof(float));
    float* ref = aligned_alloc(64, count * sizeof(float));
    
    printf("input  = %p to %p\n", (void*)input, (void*)(input + count));
    printf("output = %p to %p\n", (void*)output, (void*)(output + count));
    printf("ref    = %p to %p\n", (void*)ref, (void*)(ref + count));
    
    free(input);
    free(output);
    free(ref);
    return 0;
}
