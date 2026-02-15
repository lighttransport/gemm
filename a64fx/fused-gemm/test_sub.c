#include <stdio.h>

int main() {
    double a = 6210759.085517;
    double b = 6210759.210011;
    double c = b - a;
    
    printf("a = %.15f\n", a);
    printf("b = %.15f\n", b);
    printf("b - a = %.15f\n", c);
    
    return 0;
}
