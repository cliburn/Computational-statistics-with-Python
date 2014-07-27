#include <stdio.h>

// typedef and struct
typedef struct point {
    double x;
    double y;
    double z;
} point;

// Function declarations
void print_blank();
double array2d_sum(double **m, int r, int c);