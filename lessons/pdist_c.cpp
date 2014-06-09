#include <cmath>
#include <vector>
using namespace std;

extern "C" 

void pdist_cpp(int n, int p, vector<double> xs, vector<double> D) {
    for (int i=0; i<n; i++) {
        for (int j=0; j<n; j++) {
            double s = 0.0;
            for (int k=0; k<p; k++) {
                double tmp = xs[i*p+k] - xs[j*p+k];
                s += tmp*tmp;
            }
            D.push_back(sqrt(s));
        }
    }
}