#include <stdio.h>

int main(void)
{
    unsigned long a = 0;
    unsigned long b = 1;

    for (int i = 0; i < 10; ++i) {
        printf("%s%lu", i ? " " : "", a);
        unsigned long next = a + b;
        a = b;
        b = next;
    }
    putchar('\n');
    return 0;
}
