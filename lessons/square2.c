#include <stdio.h>
#include <math.h>

// Create a function pointer type that takes a double and returns a double
typedef double (*func)(double x);

// A higher order function that takes just such a function pointer
double apply(func f, double x)
{
    return f(x);
}

double square(double x)
{
    return x * x;
}

double cube(double x)
{
    return pow(x, 3);
}

int main()
{
    double a = 3;
    func fs[] = {square, cube, NULL};

    for (func *f=fs; *f; f++) {
        printf("%.1f\n", apply(*f, a));
    }   
}