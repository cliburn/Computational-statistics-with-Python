#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::SelfAdjointEigenSolver;

extern "C" void det(double* x, double* e, int n)
{
  MatrixXd A = MatrixXd::Map(x, n, n);
  SelfAdjointEigenSolver<MatrixXd> eigensolver(A);
  std::cout << eigensolver.eigenvalues() << "\n";
}