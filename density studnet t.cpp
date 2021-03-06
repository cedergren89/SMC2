#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <cmath>
#include <math.h>



using namespace Eigen;
using namespace std;

const double pi = 3.14159265358979323846;

double gammln(double xx);


double fun_density_t(MatrixXd x, double df, MatrixXd Mu, MatrixXd Sigma)
{
	double dim = x.rows();
	return  exp(gammln((df + dim) * 0.5)) / (exp(gammln(df*0.5)) * pow((pow((df*pi), dim)*Sigma.determinant()), 0.5)) * pow(1.0 + (1.0 / df) *((x - Mu).transpose() * Sigma.inverse() * (x - Mu))(0, 0), -(df + dim)*0.5);
}

double fun_density_t_ind(MatrixXd x, double df, MatrixXd Mu, double sigma)
{
	int dim1 = x.rows();
	MatrixXd unity(dim1, 1);
	unity.setOnes();
	MatrixXd I_inverse = unity.asDiagonal().inverse();

	double dim = x.rows();
	return  exp(gammln((df + dim) * 0.5)) / (exp(gammln(df*0.5)) * pow((pow((df*pi), dim)* pow(pow(sigma, 2), dim)), 0.5)) * pow(1.0 + (1.0 / df) *((x - Mu).transpose() * I_inverse * (x - Mu))(0, 0) * pow(sigma, (-2)), -(df + dim)*0.5);
}
