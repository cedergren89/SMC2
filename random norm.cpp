#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <ctime>

using namespace Eigen;
using namespace std;

MatrixXd fun_gen_noise(int nrolls, int dim, MatrixXd Mu, MatrixXd Sigma, char a)
{
	random_device rd;
	normal_distribution<double> distribution(0.0, 1.0);

	MatrixXd noise(nrolls, dim);
	for (int i = 0; i < nrolls; ++i) {
		for (int j = 0; j < dim; ++j) {
			noise(i, j) = distribution(rd);
		}
	}

	//choleshy decomposition
	MatrixXd Y(nrolls, dim);

	if (a == 'n')
	{
		LLT<MatrixXd> lltOfA(Sigma); // compute the Cholesky decomposition of Sigma
		MatrixXd L = lltOfA.matrixL();

		for (int i = 0; i < nrolls; ++i) {
			Y.row(i) = (Mu + L * noise.row(i).transpose()).transpose();
		}
	}

	if (a == 'y') {
		MatrixXd L = pow(Sigma.array(), 0.5);

		for (int i = 0; i < nrolls; ++i) {
			Y.row(i) = (Mu + L * noise.row(i).transpose()).transpose();
		}
	}

	return Y;
}
