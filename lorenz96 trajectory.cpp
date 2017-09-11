#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;
using namespace std;

VectorXd fun_lor96(VectorXd x, int M, double F);
MatrixXd fun_gen_noise(int nrolls, int dim, MatrixXd Mu, MatrixXd Sigma, char a);

random_device rd;
uniform_real_distribution<double> runif_03(-1,1); 
//uniform_real_distribution<double> runif_03(-2, 2);

int main()
{
	double F;
	int M;

	
	cout << "Input F:" << endl;
	cin >> F;

	cout << "Input Length of X:" << endl;
	cin >> M;

	double sigma_y = 1.0;

	VectorXd x_start(M);

	//double x0 = runif_03(rd);
	for (int i = 0; i < M; i++)
	{
		x_start(i) = runif_03(rd);
	}
	//x_start(M - 1) += 0.008;
	

	cout << "x_start" << endl;
	cout << x_start << endl;
	
	double t0, tn, delta_t;
	t0 = 0;
	tn = 5.0;
	delta_t = 0.05;

	//actual function
	int i = 0;
	double t = t0;
	int rows = (tn - t0) / delta_t + 1;
	MatrixXd xx(rows, M);
	xx.row(0) = x_start.transpose();

	VectorXd X = x_start;
	VectorXd X_new(M);
	VectorXd m1, m2, m3, m4, m;


	//input distribution
	MatrixXd Mu_input(M, 1);
	MatrixXd Sigma_input_x(M, M);
	MatrixXd Sigma_input_y(M, M);

	for (int j = 0; j < M; j++)
	{
		Mu_input(j) = 0;
	}

	double sigma_x = 0.5;
	//cout << "Insert sigma_x please:" << endl;
	//cin >> sigma_x;

	for (int jj = 0; jj < M; jj++)
	{
		for (int ii = 0; ii < M; ii++)
		{
			if (ii == jj) {
				Sigma_input_x(ii, jj) = delta_t*pow(sigma_x, 2);
			}
			else {
				Sigma_input_x(ii, jj) = 0;
			}
		}
	}

	//cout << "Independent (y/n)?" << endl;
	char a_input;
	//cin >> a_input;
	a_input = 'y';

	while (t<(tn-delta_t))
	{
		m1 = fun_lor96(X, M, F);
		m2 = fun_lor96(X + m1*delta_t / 2.0, M, F);
		m3 = fun_lor96(X + m2*delta_t / 2.0, M, F);
		m4 = fun_lor96(X + m3*delta_t, M, F);
		m = (m1 + 2 * m2 + 2 * m3 + m4) / 6;
		//m = m1;
		X_new = X + m*delta_t + fun_gen_noise(1, M, Mu_input, Sigma_input_x, a_input).row(0).transpose();
		X = X_new;
		t = t + delta_t;
		i += 1;
		xx.row(i) = X.transpose();
	}

	cout << xx << endl;
	
	

	//double sigma_y = 1.0;
	//cout << "Insert sigma_y please:" << endl;
	//cin >> sigma_y;

	for (int jj = 0; jj < M; jj++)
	{
		for (int ii = 0; ii < M; ii++)
		{
			if (ii == jj) {
				Sigma_input_y(ii, jj) = pow(sigma_y, 2);
			}
			else {
				Sigma_input_y(ii, jj) = 0;
			}
		}
	}


	
	cout << "sigmas" << endl;
	//cout << sigma_x << endl;
	//cout << Sigma_input_x << endl;
	//cout << sigma_y << endl;
	//cout << Sigma_input_y << endl;

	//adding noise 
	//MatrixXd trajectory_x = xx + fun_gen_noise(rows, M, Mu_input, Sigma_input_x, a_input);
	MatrixXd trajectory_x = xx;
	MatrixXd trajectory_y = trajectory_x + fun_gen_noise(rows, M, Mu_input, Sigma_input_y, a_input);

	//cout << trajectory_x << endl;

	ofstream fout("trajectory_y96_11.txt");
	fout << trajectory_y << endl;
	fout << endl;
	fout.close();

	//ofstream f2out("deterministic16.txt");
	//f2out << xx << endl;
	//f2out << endl;
	//f2out.close();

	ofstream f3out("trajectory_x96_11.txt");
	f3out << trajectory_x << endl;
	f3out << endl;
	f3out.close();


	system("pause");
	
	return 0;
}

