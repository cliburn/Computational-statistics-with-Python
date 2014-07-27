#include <numeric>
extern "C" double sum1d_cpp(double* x, int n)
{
  return std::accumulate(x, x+n, 0.0);
}