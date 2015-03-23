#include <math.h>

void pdist_c(int n, int p, double xs[n*p], double D[n*n]) {
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