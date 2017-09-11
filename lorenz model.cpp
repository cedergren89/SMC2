#include <Eigen/Dense>

using namespace Eigen;

VectorXd fun_lor96(VectorXd x, int M, double F)
{
	int N = M;
	//state derivatives
	VectorXd d(x.size());
	d.setZero();
	VectorXd F_vec(x.size());
	F_vec.setConstant(F);

	//edge cases, i=1,2,N
	d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0];
	d[1] = (x[2] - x[N - 1]) * x[0] - x[1];
	d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1];
	//general case
	for (int i = 2; i < N - 1; i++) {
		d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i];
	}
	d = d + F_vec;
	return d;
}
