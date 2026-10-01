#include "glm53f_lookup.h"
#include <stdio.h>
int main(void) {
    int draft[4];
    const int history[] = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    const int expected[] = {13, 14, 99, 10};
    int failed = 0;
    for (int depth = 1; depth <= 4; ++depth) {
        failed |= glm53f_lookup_draft(history, 9, depth, draft) != depth;
        for (int j = 0; j < depth; ++j) failed |= draft[j] != expected[j];
    }
    const int unique[] = {1, 2, 3, 4, 5};
    failed |= glm53f_lookup_draft(unique, 5, 4, draft) != 0;
    failed |= glm53f_lookup_draft(history, 3, 4, draft) != 0;
    failed |= glm53f_lookup_draft(history, 9, 5, draft) != 0;
    const int overlap[] = {7, 7, 7, 7};
    failed |= glm53f_lookup_draft(overlap, 4, 4, draft) != 1 || draft[0] != 7;
    const int recent[] = {1, 2, 3, 40, 1, 2, 3, 50, 1, 2, 3};
    failed |= glm53f_lookup_draft(recent, 11, 1, draft) != 1 || draft[0] != 50;
    printf("GLM53F_LOOKUP %s\n", failed ? "FAIL" : "PASS");
    return failed;
}
