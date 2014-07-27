#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

int main(int argc, char *argv[])
{
    /*
    Silly program to demonstrate some C languaage syntax 
    */

    int i = 3;
    double pi = 3.14;
    char c = 'a'; // single quotes
    char s[] = "Hello, world"; // double quotes

    printf("%s, pi=%.2f, num=%d, char=%c\n", s, pi, i, c);

    print_blank();
    
    // Initializing a struct
    point p = {.z = 2.0, .x = 3.0};
    printf("Point is (%.4f, %.4f, %.4f)\n", p.x, p.y, p.z);

    print_blank();

    // Get some numbers from command line arguments
    int n1 = atoi(argv[1]);
    int n2 = atoi(argv[2]);     

    // Using arrays with automatic memory
    int v1[n1];
    for (int i=0; i<n1; i++) {
        v1[i] = i*i;
    }

    // Using the ternary ?: operator to print either ", " or "\n"
    for (int i=0; i<n1; i++) {
        printf("%d%s", v1[i], i < (n1-1) ? ", " : "\n");
    }

    print_blank();

    // Using arrays with manual (or dynamic) memory
    double *v2 = malloc(n1 * sizeof(int));
    for (int i=0; i<n1; i++) {
        v2[i] = sin(i*i);
    }

    for (int i=0; i<n1; i++) {
        printf("%.4f%s", v2[i], i < (n1-1) ? ", " : "\n");
    }

    print_blank();
    
    // Using 2D arrays with automatic memory
    double m1[n1][n2];
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            m1[i][j] = i*n2 + j;
        }
    }

    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            printf("%8.4f%s", m1[i][j], j < (n2-1) ? ", " : "\n");
        }
    }

    print_blank();

    // Using 2D arrays with manual memory
    double **m2 = malloc(n1 * sizeof(double));
    for (int i=0; i<n1; i++) {
        m2[i] = calloc(n2, sizeof(double));
    }

    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            m2[i][j] = sqrt(i*n2 + j);
        }
    }

    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            printf("%8.4f%s", m2[i][j], j < (n2-1) ? ", " : "\n");
        }
    }

    print_blank();

    printf("The sum of entries in m2 is %.4f\n", array2d_sum(m2, n1, n2));

    // free memory after usage
    free(v2); 
    for (int i=0; i<n1; i++) {
        free(m2[i]);
    }
    free(m2);
}