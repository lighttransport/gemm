#include <stdio.h>
#include <time.h>

int main() {
    struct timespec ts1, ts2;
    
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    
    // Busy wait
    volatile double x = 0;
    for (int i = 0; i < 10000000; i++) {
        x += 1.0;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &ts2);
    
    double t1 = ts1.tv_sec + ts1.tv_nsec * 1e-9;
    double t2 = ts2.tv_sec + ts2.tv_nsec * 1e-9;
    
    printf("ts1: sec=%ld nsec=%ld\n", ts1.tv_sec, ts1.tv_nsec);
    printf("ts2: sec=%ld nsec=%ld\n", ts2.tv_sec, ts2.tv_nsec);
    printf("t1 = %f\n", t1);
    printf("t2 = %f\n", t2);
    printf("diff = %f seconds\n", t2 - t1);
    
    return 0;
}
