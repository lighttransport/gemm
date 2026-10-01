#ifndef GLM53F_LOOKUP_H
#define GLM53F_LOOKUP_H
#include <string.h>
/* Longest suffix first, most recent match first. A draft contains only tokens
 * already observed in this request; overlapping occurrences are valid guesses. */
static inline int glm53f_lookup_draft(const int *history, int count,
                                      int depth, int draft[4]) {
    if (!history || !draft || count < 4 || depth < 1 || depth > 4) return 0;
    int longest = count - 1 < 8 ? count - 1 : 8;
    for (int order = longest; order >= 3; --order) {
        const int *suffix = history + count - order;
        for (int start = count - order - 1; start >= 0; --start) {
            if (memcmp(history + start, suffix, (size_t)order * sizeof(int))) continue;
            int n = count - start - order;
            if (n > depth) n = depth;
            memcpy(draft, history + start + order, (size_t)n * sizeof(int));
            return n;
        }
    }
    return 0;
}
#endif
