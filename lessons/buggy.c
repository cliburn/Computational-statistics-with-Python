
# Create a function pointer type that takes a double and returns a double
double *func(double x);

# A higher order function that takes just such a function pointer
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
    return pow(3, x);
}

double mystery(double x)
{
    double y = 10;
    if (x < 10)
        x = square(x);
    else
        x += y;
        x = cube(x);
    return x;
}

int main()
{
    double a = 3;
    func fs[] = {square, cube, mystery, NULL}

    for (func *f=fs, f != NULL, f++) {
        printf("%d\n", apply(f, a));
    }   
}