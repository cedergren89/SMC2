#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;
using namespace std;

const double pi = 3.14159265358979323846;

MatrixXd fun_mvdnorm(MatrixXd x, MatrixXd Mu, MatrixXd Sigma)
{
	int dim = x.rows();
	MatrixXd prob = 1 / pow(pow(2 * pi, dim) * Sigma.determinant() * ((x - Mu).transpose()*Sigma.inverse()*(x - Mu)).array().exp(), 0.5);
	return prob;
}

MatrixXd fun_mvdnorm_ind(MatrixXd x, MatrixXd Mu, double sigma)
{
	int dim = x.rows();
	MatrixXd unity(dim, 1);
	unity.setOnes();
	MatrixXd I_inverse = unity.asDiagonal().inverse();
	MatrixXd prob = 1 / pow(pow(2 * pi * pow(sigma, 2), dim) * ((x - Mu).transpose()*I_inverse*(x - Mu) * pow(sigma, (-2))).array().exp(), 0.5);

	return prob;
}
