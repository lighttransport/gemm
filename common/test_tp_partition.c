#include <assert.h>
#include "tp_partition.h"

static void check(int n, int parts, int alignment, const int *expected) {
    int cursor = 0;
    for (int rank = 0; rank < parts; rank++) {
        int lo, hi;
        tf_tp_range_aligned(n, parts, rank, alignment, &lo, &hi);
        assert(lo == cursor);
        assert(hi >= lo && hi <= n);
        if (rank + 1 < parts) assert((hi - lo) % alignment == 0);
        assert(hi - lo == expected[rank]);
        cursor = hi;
    }
    assert(cursor == n);
}

int main(void) {
    const int tp6[]  = {3072, 3072, 2816, 2816, 2816, 2816};
    const int tp8[]  = {2304, 2304, 2304, 2304, 2048, 2048, 2048, 2048};
    const int tp10[] = {1792, 1792, 1792, 1792, 1792, 1792, 1792, 1792, 1536, 1536};
    const int tp12[] = {1536, 1536, 1536, 1536, 1536, 1536, 1536, 1536,
                        1280, 1280, 1280, 1280};
    check(17408, 6, 256, tp6);
    check(17408, 8, 256, tp8);
    check(17408, 10, 256, tp10);
    check(17408, 12, 256, tp12);
    const int partial[] = {512, 512, 176};
    check(1200, 3, 256, partial);
    return 0;
}
