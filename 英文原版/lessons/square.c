#include <stdio.h>

double square(double x)
{
    return x * x;
}

int main()
{
    double a = 3;
    printf("%f\n", square(a));
}