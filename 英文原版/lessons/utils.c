#include "utils.h"

// Function definitions
void print_blank() {
    printf("\n");
}

double array2d_sum(double **m, int r, int c) {
    double s= 0.0;
    for (int i=0; i<r; i++) {
        for (int j=0; j<c; j++) {
            s += m[i][j];
        }
    }
    return s;
}