#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd fun_lor96(VectorXd x, int M, double F);
MatrixXd fun_gen_noise(int nrolls, int dim, MatrixXd Mu, MatrixXd Sigma, char a);

MatrixXd fun_prop_det(double F, int M, VectorXd x_start, double h)
{
	VectorXd X = x_start;
	VectorXd X_new(M);
	VectorXd m1, m2, m3, m4, m;

	m1 = fun_lor96(X, M, F);
	m2 = fun_lor96(X + m1*h / 2.0, M, F);
	m3 = fun_lor96(X + m2*h / 2.0, M, F);
	m4 = fun_lor96(X + m3*h, M, F);
	m = (m1 + 2 * m2 + 2 * m3 + m4) / 6;
	X_new = X + m*h;
	X = X_new;
	return X.transpose();
}