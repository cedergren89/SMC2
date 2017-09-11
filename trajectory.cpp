#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd fun_lor96(VectorXd x, int M, double F);
MatrixXd fun_gen_noise(int nrolls, int dim, MatrixXd Mu, MatrixXd Sigma, char a);

MatrixXd fun_trajectory(double F, int M, VectorXd x_start, double t0, double tn, double h, MatrixXd Mu_input, MatrixXd Sigma_input_x, MatrixXd Sigma_input_y, char ind)
{
	//actual function
	int i = 0;
	double t = t0;
	int rows = (tn - t0) / h + 2;
	MatrixXd xx(rows, M);
	xx.row(0) = x_start.transpose();

	VectorXd X = x_start;
	VectorXd m1, m2, m3, m4, m;

	while (t<tn)
	{
		m1 = fun_lor96(X, M, F);
		m2 = fun_lor96(X + m1*h / 2.0, M, F);
		m3 = fun_lor96(X + m2*h / 2.0, M, F);
		m4 = fun_lor96(X + m3*h, M, F);
		m = (m1 + 2 * m2 + 2 * m3 + m4) / 6;
		X = X + m*h;
		t = t + h;
		i += 1;
		xx.row(i) = X.transpose();
	}

	//adding noise 
	MatrixXd trajectory_x = xx + fun_gen_noise(rows, M, Mu_input, Sigma_input_x, ind);
	MatrixXd trajectory_y = trajectory_x + fun_gen_noise(rows, M, Mu_input, Sigma_input_y, ind);

	//	ofstream fout("trajectory.txt");
	//	fout << trajectory << endl;
	//	fout << endl;
	//	fout.close();

	ofstream f2out("deterministic.txt");
	f2out << trajectory_x << endl;
	f2out << endl;
	f2out.close();

	return trajectory_y;
}

