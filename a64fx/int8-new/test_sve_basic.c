#include <stdio.h>
#include <stdlib.h>
#include <arm_sve.h>

int main(void) {
    printf("Testing basic SVE operations...\n");

    const size_t N = 64;
    int8_t* data = (int8_t*)malloc(N * sizeof(int8_t));

    for (size_t i = 0; i < N; i++) {
        data[i] = i;
    }

    printf("Data initialized\n");

    // Try basic SVE load
    svbool_t pg = svptrue_b8();
    svint8_t v = svld1_s8(pg, data);

    printf("SVE load successful\n");

    // Try unpacking
    svint16_t v16 = svunpklo_s16(v);
    printf("SVE unpack successful\n");

    svint32_t v32 = svunpklo_s32(v16);
    printf("SVE unpack to INT32 successful\n");

    // Try addition
    svint32_t sum_vec = svdup_n_s32(0);
    sum_vec = svadd_s32_x(pg, sum_vec, v32);
    printf("SVE add successful\n");

    // Try reduction
    svbool_t pg32 = svptrue_b32();
    int32_t sum = svaddv_s32(pg32, sum_vec);
    printf("SVE reduction successful, sum=%d\n", sum);

    free(data);

    printf("✓ All SVE operations passed\n");
    return 0;
}
