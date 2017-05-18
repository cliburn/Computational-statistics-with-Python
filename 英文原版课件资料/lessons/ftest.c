#include <apop.h>

double ftest(double n1, double n2, double n3, double n4)
{
  apop_data *testdata = apop_data_fill(apop_data_alloc(2,2),
                                       n1, n2,
                                       n3, n4);
  testdata = apop_test_fisher_exact(testdata);
  double p = apop_data_get(testdata, 1, -1);
  apop_data_free(testdata);
  return p;
}