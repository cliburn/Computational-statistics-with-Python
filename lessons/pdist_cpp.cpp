#include <cmath>

extern "C" 

// Variable length arrays are OK for C99 but not legal in C++
// void pdist_cpp(int n, int p, double xs[n*p], double D[n*n]) {
void pdist_cpp(int n, int p, double *xs, double *D) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<p; k++) {
                double tmp = xs[i*p+k] - xs[j*p+k];
                s += tmp*tmp;
            }
            D[i*n+j] = sqrt(s);
        }
    }
}