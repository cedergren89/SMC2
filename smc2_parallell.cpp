#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <random>
#include <ctime>
#include <list>
#include <omp.h>

#include <numeric>


using namespace std;
using namespace Eigen;


MatrixXd fun_trajectory(double F, int M, VectorXd x_start, double t0, double tn, double h, MatrixXd Mu_input, MatrixXd Sigma_input_x, MatrixXd Sigma_input_y, char ind);
MatrixXd fun_prop_det(double F, int M, VectorXd x_start, double h);
MatrixXd fun_gen_noise(int nrolls, int dim, MatrixXd Mu, MatrixXd Sigma, char a); //redundant??
MatrixXd fun_mvdnorm(MatrixXd x, MatrixXd Mu, MatrixXd Sigma);
MatrixXd fun_mvdnorm_ind(MatrixXd x, MatrixXd Mu, double sigma);
double fun_logdinvgamma(double x, double a, double b);
double fun_density_t_ind(MatrixXd x, double df, MatrixXd Mu, double sigma);

double fun_ess_r(double phi_prop, double phi_prev, MatrixXd weights)
{
	double value = 1 / (pow(pow(weights.array(), (phi_prop - phi_prev)) / (pow(weights.array(), (phi_prop - phi_prev))).sum(), 2)).sum();
	return value;
}

int main()
{
	time_t start, finish;
	start = time(0);

	//Set parameters	
	int T = 11;
	double delta_t = 0.05;

	int M = 96;

	int N_theta = 400;
	int N_x = 2000;

	double alpha_temp = 0.5;
	int nsteps = 5;
	//double rho = 0.0001;  	//when gap is 4
	//double rho = 0.5;		//when prior is 1,1
	double rho = 0.99;		//when prior is 1,1/3
	double df = 1;

	double gap_double = 1.0;	

	uniform_real_distribution<double> runif_03(-1,1);
	
	MatrixXd X_prop(N_x, M);
	double F_in_proposal;
	double tn_in_proposal;

	VectorXd t_seq(101);
	t_seq(0) = 0;
	for (int i = 1; i < 101; i++)
	{
		t_seq(i) = t_seq(i - 1) + delta_t;
	}

	//Import trajectory_y
	MatrixXd trajectory_y(T, M);

	ifstream myfile;
	myfile.open("trajectory_y96_11.txt", ios::in);

	string line;
	int j = 0;

	double a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36;
	double a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64;
	double a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99, a100;
	double a101, a102, a103, a104, a105, a106, a107, a108, a109, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a120, a121, a122, a123, a124, a125, a126, a127, a128;

	if (myfile.is_open())
	{
		for (int i = 0; i < T; i++)
		{
			if (M == 4)
			{
				myfile >> a1 >> a2 >> a3 >> a4;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
			}

			if (M == 8)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
			}

			if (M == 16)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9 >> a10 >> a11 >> a12 >> a13 >> a14 >> a15 >> a16;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
				trajectory_y(j, 8) = a9;
				trajectory_y(j, 9) = a10;
				trajectory_y(j, 10) = a11;
				trajectory_y(j, 11) = a12;
				trajectory_y(j, 12) = a13;
				trajectory_y(j, 13) = a14;
				trajectory_y(j, 14) = a15;
				trajectory_y(j, 15) = a16;
			}
			
			if (M == 32)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9 >> a10 >> a11 >> a12 >> a13 >> a14 >> a15 >> a16 >> a17 >> a18 >> a19 >> a20 >> a21 >> a22 >> a23 >> a24 >> a25 >> a26 >> a27 >> a28 >> a29 >> a30 >> a31 >> a32;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
				trajectory_y(j, 8) = a9;
				trajectory_y(j, 9) = a10;
				trajectory_y(j, 10) = a11;
				trajectory_y(j, 11) = a12;
				trajectory_y(j, 12) = a13;
				trajectory_y(j, 13) = a14;
				trajectory_y(j, 14) = a15;
				trajectory_y(j, 15) = a16;
				trajectory_y(j, 16) = a17;
				trajectory_y(j, 17) = a18;
				trajectory_y(j, 18) = a19;
				trajectory_y(j, 19) = a20;
				trajectory_y(j, 20) = a21;
				trajectory_y(j, 21) = a22;
				trajectory_y(j, 22) = a23;
				trajectory_y(j, 23) = a24;
				trajectory_y(j, 24) = a25;
				trajectory_y(j, 25) = a26;
				trajectory_y(j, 26) = a27;
				trajectory_y(j, 27) = a28;
				trajectory_y(j, 28) = a29;
				trajectory_y(j, 29) = a30;
				trajectory_y(j, 30) = a31;
				trajectory_y(j, 31) = a32;
				//trajectory_y(j, 32) = a33;
				//trajectory_y(j, 33) = a34;
				//trajectory_y(j, 34) = a35;
				//trajectory_y(j, 35) = a36;
			}

			if (M == 64)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9 >> a10 >> a11 >> a12 >> a13 >> a14 >> a15 >> a16 >> a17 >> a18 >> a19 >> a20 >> a21 >> a22 >> a23 >> a24 >> a25 >> a26 >> a27 >> a28 >> a29 >> a30 >> a31 >> a32 >> a33 >> a34 >> a35 >> a36 >> a37 >> a38 >> a39 >> a40 >> a41 >> a42 >> a43 >> a44 >> a45 >> a46 >> a47 >> a48 >> a49 >> a50 >> a51 >> a52 >> a53 >> a54 >> a55 >> a56 >> a57 >> a58 >> a59 >> a60 >> a61 >> a62 >> a63 >> a64;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
				trajectory_y(j, 8) = a9;
				trajectory_y(j, 9) = a10;
				trajectory_y(j, 10) = a11;
				trajectory_y(j, 11) = a12;
				trajectory_y(j, 12) = a13;
				trajectory_y(j, 13) = a14;
				trajectory_y(j, 14) = a15;
				trajectory_y(j, 15) = a16;
				trajectory_y(j, 16) = a17;
				trajectory_y(j, 17) = a18;
				trajectory_y(j, 18) = a19;
				trajectory_y(j, 19) = a20;
				trajectory_y(j, 20) = a21;
				trajectory_y(j, 21) = a22;
				trajectory_y(j, 22) = a23;
				trajectory_y(j, 23) = a24;
				trajectory_y(j, 24) = a25;
				trajectory_y(j, 25) = a26;
				trajectory_y(j, 26) = a27;
				trajectory_y(j, 27) = a28;
				trajectory_y(j, 28) = a29;
				trajectory_y(j, 29) = a30;
				trajectory_y(j, 30) = a31;
				trajectory_y(j, 31) = a32;
				trajectory_y(j, 32) = a33;
				trajectory_y(j, 33) = a34;
				trajectory_y(j, 34) = a35;
				trajectory_y(j, 35) = a36;
				trajectory_y(j, 36) = a37;
				trajectory_y(j, 37) = a38;
				trajectory_y(j, 38) = a39;
				trajectory_y(j, 39) = a40;
				trajectory_y(j, 40) = a41;
				trajectory_y(j, 41) = a42;
				trajectory_y(j, 42) = a43;
				trajectory_y(j, 43) = a44;
				trajectory_y(j, 44) = a45;
				trajectory_y(j, 45) = a46;
				trajectory_y(j, 46) = a47;
				trajectory_y(j, 47) = a48;
				trajectory_y(j, 48) = a49;
				trajectory_y(j, 49) = a50;
				trajectory_y(j, 50) = a51;
				trajectory_y(j, 51) = a52;
				trajectory_y(j, 52) = a53;
				trajectory_y(j, 53) = a54;
				trajectory_y(j, 54) = a55;
				trajectory_y(j, 55) = a56;
				trajectory_y(j, 56) = a57;
				trajectory_y(j, 57) = a58;
				trajectory_y(j, 58) = a59;
				trajectory_y(j, 59) = a60;
				trajectory_y(j, 60) = a61;
				trajectory_y(j, 61) = a62;
				trajectory_y(j, 62) = a63;
				trajectory_y(j, 63) = a64;
			}

			if (M == 96)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9 >> a10 >> a11 >> a12 >> a13 >> a14 >> a15 >> a16 >> a17 >> a18 >> a19 >> a20 >> a21 >> a22 >> a23 >> a24 >> a25 >> a26 >> a27 >> a28 >> a29 >> a30 >> a31 >> a32 >> a33 >> a34 >> a35 >> a36 >> a37 >> a38 >> a39 >> a40 >> a41 >> a42 >> a43 >> a44 >> a45 >> a46 >> a47 >> a48 >> a49 >> a50 >> a51 >> a52 >> a53 >> a54 >> a55 >> a56 >> a57 >> a58 >> a59 >> a60 >> a61 >> a62 >> a63 >> a64 >> a65 >> a66 >> a67 >> a68 >> a69 >> a70 >> a71 >> a72 >> a73 >> a74 >> a75 >> a76 >> a77 >> a78 >> a79 >> a80 >> a81 >> a82 >> a83 >> a84 >> a85 >> a86 >> a87 >> a88 >> a89 >> a90 >> a91 >> a92 >> a93 >> a94 >> a95 >> a96;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
				trajectory_y(j, 8) = a9;
				trajectory_y(j, 9) = a10;
				trajectory_y(j, 10) = a11;
				trajectory_y(j, 11) = a12;
				trajectory_y(j, 12) = a13;
				trajectory_y(j, 13) = a14;
				trajectory_y(j, 14) = a15;
				trajectory_y(j, 15) = a16;
				trajectory_y(j, 16) = a17;
				trajectory_y(j, 17) = a18;
				trajectory_y(j, 18) = a19;
				trajectory_y(j, 19) = a20;
				trajectory_y(j, 20) = a21;
				trajectory_y(j, 21) = a22;
				trajectory_y(j, 22) = a23;
				trajectory_y(j, 23) = a24;
				trajectory_y(j, 24) = a25;
				trajectory_y(j, 25) = a26;
				trajectory_y(j, 26) = a27;
				trajectory_y(j, 27) = a28;
				trajectory_y(j, 28) = a29;
				trajectory_y(j, 29) = a30;
				trajectory_y(j, 30) = a31;
				trajectory_y(j, 31) = a32;
				trajectory_y(j, 32) = a33;
				trajectory_y(j, 33) = a34;
				trajectory_y(j, 34) = a35;
				trajectory_y(j, 35) = a36;
				trajectory_y(j, 36) = a37;
				trajectory_y(j, 37) = a38;
				trajectory_y(j, 38) = a39;
				trajectory_y(j, 39) = a40;
				trajectory_y(j, 40) = a41;
				trajectory_y(j, 41) = a42;
				trajectory_y(j, 42) = a43;
				trajectory_y(j, 43) = a44;
				trajectory_y(j, 44) = a45;
				trajectory_y(j, 45) = a46;
				trajectory_y(j, 46) = a47;
				trajectory_y(j, 47) = a48;
				trajectory_y(j, 48) = a49;
				trajectory_y(j, 49) = a50;
				trajectory_y(j, 50) = a51;
				trajectory_y(j, 51) = a52;
				trajectory_y(j, 52) = a53;
				trajectory_y(j, 53) = a54;
				trajectory_y(j, 54) = a55;
				trajectory_y(j, 55) = a56;
				trajectory_y(j, 56) = a57;
				trajectory_y(j, 57) = a58;
				trajectory_y(j, 58) = a59;
				trajectory_y(j, 59) = a60;
				trajectory_y(j, 60) = a61;
				trajectory_y(j, 61) = a62;
				trajectory_y(j, 62) = a63;
				trajectory_y(j, 63) = a64;
				trajectory_y(j, 64) = a65;
				trajectory_y(j, 65) = a66;
				trajectory_y(j, 66) = a67;
				trajectory_y(j, 67) = a68;
				trajectory_y(j, 68) = a69;
				trajectory_y(j, 69) = a70;
				trajectory_y(j, 70) = a71;
				trajectory_y(j, 71) = a72;
				trajectory_y(j, 72) = a73;
				trajectory_y(j, 73) = a74;
				trajectory_y(j, 74) = a75;
				trajectory_y(j, 75) = a76;
				trajectory_y(j, 76) = a77;
				trajectory_y(j, 77) = a78;
				trajectory_y(j, 78) = a79;
				trajectory_y(j, 79) = a80;
				trajectory_y(j, 80) = a81;
				trajectory_y(j, 81) = a82;
				trajectory_y(j, 82) = a83;
				trajectory_y(j, 83) = a84;
				trajectory_y(j, 84) = a85;
				trajectory_y(j, 85) = a86;
				trajectory_y(j, 86) = a87;
				trajectory_y(j, 87) = a88;
				trajectory_y(j, 88) = a89;
				trajectory_y(j, 89) = a90;
				trajectory_y(j, 90) = a91;
				trajectory_y(j, 91) = a92;
				trajectory_y(j, 92) = a93;
				trajectory_y(j, 93) = a94;
				trajectory_y(j, 94) = a95;
				trajectory_y(j, 95) = a96;
			}
			
			if (M == 128)
			{
				myfile >> a1 >> a2 >> a3 >> a4 >> a5 >> a6 >> a7 >> a8 >> a9 >> a10 >> a11 >> a12 >> a13 >> a14 >> a15 >> a16 >> a17 >> a18 >> a19 >> a20 >> a21 >> a22 >> a23 >> a24 >> a25 >> a26 >> a27 >> a28 >> a29 >> a30 >> a31 >> a32 >> a33 >> a34 >> a35 >> a36 >> a37 >> a38 >> a39 >> a40 >> a41 >> a42 >> a43 >> a44 >> a45 >> a46 >> a47 >> a48 >> a49 >> a50 >> a51 >> a52 >> a53 >> a54 >> a55 >> a56 >> a57 >> a58 >> a59 >> a60 >> a61 >> a62 >> a63 >> a64 >> a65 >> a66 >> a67 >> a68 >> a69 >> a70 >> a71 >> a72 >> a73 >> a74 >> a75 >> a76 >> a77 >> a78 >> a79 >> a80 >> a81 >> a82 >> a83 >> a84 >> a85 >> a86 >> a87 >> a88 >> a89 >> a90 >> a91 >> a92 >> a93 >> a94 >> a95 >> a96 >> a97 >> a98 >> a99 >> a100 >> a101 >> a102 >> a103 >> a104 >> a105 >> a106 >> a107 >> a108 >> a109 >> a110 >> a111 >> a112 >> a113 >> a114 >> a115 >> a116 >> a117 >> a118 >> a119 >> a120 >> a121 >> a122 >> a123 >> a124 >> a125 >> a126 >> a127 >> a128;

				trajectory_y(j, 0) = a1;
				trajectory_y(j, 1) = a2;
				trajectory_y(j, 2) = a3;
				trajectory_y(j, 3) = a4;
				trajectory_y(j, 4) = a5;
				trajectory_y(j, 5) = a6;
				trajectory_y(j, 6) = a7;
				trajectory_y(j, 7) = a8;
				trajectory_y(j, 8) = a9;
				trajectory_y(j, 9) = a10;
				trajectory_y(j, 10) = a11;
				trajectory_y(j, 11) = a12;
				trajectory_y(j, 12) = a13;
				trajectory_y(j, 13) = a14;
				trajectory_y(j, 14) = a15;
				trajectory_y(j, 15) = a16;
				trajectory_y(j, 16) = a17;
				trajectory_y(j, 17) = a18;
				trajectory_y(j, 18) = a19;
				trajectory_y(j, 19) = a20;
				trajectory_y(j, 20) = a21;
				trajectory_y(j, 21) = a22;
				trajectory_y(j, 22) = a23;
				trajectory_y(j, 23) = a24;
				trajectory_y(j, 24) = a25;
				trajectory_y(j, 25) = a26;
				trajectory_y(j, 26) = a27;
				trajectory_y(j, 27) = a28;
				trajectory_y(j, 28) = a29;
				trajectory_y(j, 29) = a30;
				trajectory_y(j, 30) = a31;
				trajectory_y(j, 31) = a32;
				trajectory_y(j, 32) = a33;
				trajectory_y(j, 33) = a34;
				trajectory_y(j, 34) = a35;
				trajectory_y(j, 35) = a36;
				trajectory_y(j, 36) = a37;
				trajectory_y(j, 37) = a38;
				trajectory_y(j, 38) = a39;
				trajectory_y(j, 39) = a40;
				trajectory_y(j, 40) = a41;
				trajectory_y(j, 41) = a42;
				trajectory_y(j, 42) = a43;
				trajectory_y(j, 43) = a44;
				trajectory_y(j, 44) = a45;
				trajectory_y(j, 45) = a46;
				trajectory_y(j, 46) = a47;
				trajectory_y(j, 47) = a48;
				trajectory_y(j, 48) = a49;
				trajectory_y(j, 49) = a50;
				trajectory_y(j, 50) = a51;
				trajectory_y(j, 51) = a52;
				trajectory_y(j, 52) = a53;
				trajectory_y(j, 53) = a54;
				trajectory_y(j, 54) = a55;
				trajectory_y(j, 55) = a56;
				trajectory_y(j, 56) = a57;
				trajectory_y(j, 57) = a58;
				trajectory_y(j, 58) = a59;
				trajectory_y(j, 59) = a60;
				trajectory_y(j, 60) = a61;
				trajectory_y(j, 61) = a62;
				trajectory_y(j, 62) = a63;
				trajectory_y(j, 63) = a64;
				trajectory_y(j, 64) = a65;
				trajectory_y(j, 65) = a66;
				trajectory_y(j, 66) = a67;
				trajectory_y(j, 67) = a68;
				trajectory_y(j, 68) = a69;
				trajectory_y(j, 69) = a70;
				trajectory_y(j, 70) = a71;
				trajectory_y(j, 71) = a72;
				trajectory_y(j, 72) = a73;
				trajectory_y(j, 73) = a74;
				trajectory_y(j, 74) = a75;
				trajectory_y(j, 75) = a76;
				trajectory_y(j, 76) = a77;
				trajectory_y(j, 77) = a78;
				trajectory_y(j, 78) = a79;
				trajectory_y(j, 79) = a80;
				trajectory_y(j, 80) = a81;
				trajectory_y(j, 81) = a82;
				trajectory_y(j, 82) = a83;
				trajectory_y(j, 83) = a84;
				trajectory_y(j, 84) = a85;
				trajectory_y(j, 85) = a86;
				trajectory_y(j, 86) = a87;
				trajectory_y(j, 87) = a88;
				trajectory_y(j, 88) = a89;
				trajectory_y(j, 89) = a90;
				trajectory_y(j, 90) = a91;
				trajectory_y(j, 91) = a92;
				trajectory_y(j, 92) = a93;
				trajectory_y(j, 93) = a94;
				trajectory_y(j, 94) = a95;
				trajectory_y(j, 95) = a96;
				trajectory_y(j, 96) = a97;
				trajectory_y(j, 97) = a98;
				trajectory_y(j, 98) = a99;
				trajectory_y(j, 99) = a100;
				trajectory_y(j, 100) = a101;
				trajectory_y(j, 101) = a102;
				trajectory_y(j, 102) = a103;
				trajectory_y(j, 103) = a104;
				trajectory_y(j, 104) = a105;
				trajectory_y(j, 105) = a106;
				trajectory_y(j, 106) = a107;
				trajectory_y(j, 107) = a108;
				trajectory_y(j, 108) = a109;
				trajectory_y(j, 109) = a110;
				trajectory_y(j, 110) = a111;
				trajectory_y(j, 111) = a112;
				trajectory_y(j, 112) = a113;
				trajectory_y(j, 113) = a114;
				trajectory_y(j, 114) = a115;
				trajectory_y(j, 115) = a116;
				trajectory_y(j, 116) = a117;
				trajectory_y(j, 117) = a118;
				trajectory_y(j, 118) = a119;
				trajectory_y(j, 119) = a120;
				trajectory_y(j, 120) = a121;
				trajectory_y(j, 121) = a122;
				trajectory_y(j, 122) = a123;
				trajectory_y(j, 123) = a124;
				trajectory_y(j, 124) = a125;
				trajectory_y(j, 125) = a126;
				trajectory_y(j, 126) = a127;
				trajectory_y(j, 127) = a128;
			}

			j++;
		}
		myfile.close();
	}
	else
	{
		cout << "Could not open file" << endl;
	}
	//Import trajectory END
	
	//partial declarations
	int gap = gap_double;

	VectorXi sequence(M);
	for (int i = 0; i < M; i++)
	{
		sequence(i) = i;
	}
	double M_double = M;
	double M_partial_double = M_double / gap_double;
	int M_partial = M_partial_double;
	VectorXi partial_sequence(M_partial);
	for (int i = 0; i < M_partial; i++)
	{
		partial_sequence(i) = i*gap;
	}
	MatrixXd trajectory_y_partial(T, M_partial);
	for (int i = 0; i < M_partial; i++)
	{
		trajectory_y_partial.col(i) = trajectory_y.col(partial_sequence(i));
	}

	MatrixXd X_partial(N_x, M_partial);

	int T_a = T;

	//Initialise particles
	double N_theta_double = N_theta;
	VectorXd F(N_theta);
	VectorXd sigma(N_theta);

	random_device rd;
	double p1 = 6;
	double p2 = 11;
	uniform_real_distribution<double> runif_F(p1, p2);
	
	double p3 = 1.0;
	double p4 = 3.0;
	gamma_distribution<double> rgamma(p3, p4);      
													 
	uniform_real_distribution<double> runif_01(0, 1);
	uniform_real_distribution<double> runif_sigma0(0, 3);
	

	for (int i = 0; i < N_theta; i++)
	{
		F(i) = runif_F(rd);
		sigma(i) = 1.0 / rgamma(rd);	
	}
	//Initialise particles END

	VectorXd F_init = F;
	VectorXd sigma_init = sigma;

	//input N,M
	double N_x_double = N_x;
	double sigma_x = 0.5;
	double sigma_y = 1.0;
	
	char ind = 'y';
	
	//Mean and covariance initialisation
	MatrixXd Mu_0(M, 1);		//for initialising step of X particles
	MatrixXd Mu_prop(M, 1);		//for proposal step of X particles
	Mu_prop.setZero();

	MatrixXd Sigma_y(M, M);
	MatrixXd Sigma_prop(M, M);

	for (int j = 0; j < M; j++)
	{
		break;
		for (int i = 0; i < M; i++)
		{
			if (i == j) {
				Sigma_y(i, j) = sigma_y;
				Sigma_prop(i, j) = delta_t*pow(sigma_x, 2);			
			}
			else {
				Sigma_y(i, j) = 0;
				Sigma_prop(i, j) = 0;
			}
		}
	}

	//resampling
	int index;
	VectorXd W_cum(N_x);
	MatrixXd W_theta_cum(N_theta, 1);
	VectorXd U(N_x);
	VectorXd U_theta(N_theta);
	double U1;
	double i_;
	double j_;
	VectorXi indices(N_x);
	VectorXi indices_theta(N_theta);
	vector<MatrixXd> X_array;
	vector<MatrixXd> X_array_temp;

	//Initialise X_0 (NEW)
	MatrixXd X_0(N_x, M);

	for (int j = 0; j < N_theta; j++)
	{
		//randomly initialising mean vector based on prior
		for (int i = 0; i < N_x; i++)
		{
			for (int m = 0; m < M; m++)
			{
				X_0(i, m) = runif_03(rd);
			}
		}
		X_array.push_back(X_0);
	}

	cout << "ok initialsing " << endl;
	//Initialise weights
	MatrixXd w_x(N_x, N_theta);
	for (int j = 0; j < N_theta; j++)
	{
		for (int i = 0; i < N_x; i++)
		{
			w_x(i, j) = 1.0 / N_x_double;
		}
	}
	
	VectorXd W_x(N_x);
	MatrixXd w_theta(N_theta, 1);
	MatrixXd W_theta(N_theta, 1);
	for (int i = 0; i < N_theta; i++)
	{
		w_theta(i, 0) = 1.0 / N_x_double;
		W_theta(i, 0) = 1.0 / N_theta_double;
	}
	
	MatrixXd likelihood_acc = w_theta;
	MatrixXd likelihood_now = w_theta;
	MatrixXd likelihood_divided(N_theta, 2);
	MatrixXd likelihood_acc_temp = w_theta;
	VectorXd W_theta_prev = W_theta;

	double ess_theta_r = 1 / pow(W_theta.array(), 2).sum(); 
	double L_r = 1;

	//Monitoring
	vector<int> k_path(1);
	vector<int> r_path(1);
	vector<double> phi_r_path(1);
	vector<double> ess_theta_r_path(1);
	vector<double> L_r_path(1);
	vector<int> count(1);
	vector<double> phi_one(1);

	k_path[0] = 0;
	r_path[0] = 1;
	phi_r_path[0] = 1;
	ess_theta_r_path[0] = N_theta;
	L_r_path[0] = N_theta;
	count[0] = 0;
	phi_one[0] = 0;

	//tempering
	int r;
	double phi_r;
	double phi_prev;
	double a;
	double b;
	double c;
	double prec = 0.001;
	
	//VectorXd cooling(3);
	//cooling << 1.0 / 32.0, 1.0 / 4.0, 1.0;

	//compute moments
	double mean_F;
	double mean_sigma;
	MatrixXd mean_prop(2, 1);
	MatrixXd para_matrix(N_theta, 2);
	MatrixXd cov_prop(2, 2);
	cov_prop.fill(0);

	//Resample prep
	VectorXd F_temp(N_theta);
	VectorXd sigma_temp(N_theta);
	MatrixXd likelihood_now_temp(N_theta, 1);
	MatrixXd likelihood_divided_temp(N_theta, 2);

	//MCMC rejuvenation prep
	double nsteps_double = nsteps;
	int nsteps_1 = nsteps + 1;
	VectorXd F_mcmc(nsteps_1);
	VectorXd sigma_mcmc(nsteps_1);
	MatrixXd prev_mat(2, 1);
	MatrixXd prev_weights(N_x, N_theta);

	MatrixXd Mu_noise_mat(2, 1);
	Mu_noise_mat(0, 0) = 0;
	Mu_noise_mat(1, 0) = 0;
	Mu_noise_mat.setZero();
	
	MatrixXd prev_logl_divided(2, 1);

	MatrixXd Mu_mcmc(2, 1);
	Mu_mcmc.fill(0);
	MatrixXd noise_mat(2, 1);
	MatrixXd prop_mat(2, 1);

	char dep = 'n';
	
	MatrixXd logl_divided_pf(2, 1);
	logl_divided_pf.fill(0);

	VectorXd acc_mcmc(nsteps);
	MatrixXd Sigma_mcmc(2, 2);

	double F_prop;
	double sigma_prop;

	double logl;
	double logl_prior;
	double logl_prop;
	double prev_logl;
	double prev_logl_prior;
	double rev_logl_prop;
	double r1;
	double r2;
	double alpha_ratio;

	//Monitoring MCMC
	double acc_r = 0;
	double T_steps = 0;
	double T_acc = 0;
	vector<double> acc_ratio_r_path(1);
	acc_ratio_r_path[0] = 0; 
	vector<double> acc_append;

	vector<VectorXd> F_before;
	vector<VectorXd> F_after;
	vector<VectorXd> sigma_before;
	vector<VectorXd> sigma_after;

	double J_F_r;
	double J_sigma_r;
	vector<double> J_F_path(1);
	vector<double> J_sigma_path(1);
-	J_F_path[0] = 0; 
	J_sigma_path[0] = 0; 

	vector<MatrixXd> moves_F(nsteps);
	vector<MatrixXd> moves_sigma(nsteps);
	vector<MatrixXd> moves_F_array;
	vector<MatrixXd> moves_sigma_array;

	//Storing theta particles at time points k
	MatrixXd F_path(N_theta, T_a);
	MatrixXd sigma_path(N_theta, T_a);

	//Particle filter prep
	int T_pf;

	MatrixXd Mu_0_pf(M, 1);
	MatrixXd Sigma_prop_pf(M, M);

	MatrixXd X_pf(N_x, M);
	VectorXd X_prev_pf(M);
	MatrixXd X_temp_pf(N_x, M);

	MatrixXd noise_pf(N_x, M);

	double k_prev = 0;
	double alpha_pf = 1;
	double j_res_pf;
	
	MatrixXd w_temp_pf(N_x, 1);
	MatrixXd w_pf(N_x, 1);
	MatrixXd W_pf(N_x, 1);
	MatrixXd W_cum_pf(N_x, 1);

	//size dependent
	VectorXd likelihood_pf(T);	
	VectorXd logl_pf(T);
	logl_pf.fill(0);
	VectorXd ess_pf(T);
	VectorXi k_res_pf(T);
	MatrixXd X_bar_pf(T, M);
	ess_pf.setZero();
	k_res_pf.setZero();

	for (int k = 0; k < T_a; k++)
	{
		if (k > 0)
		{
			//resample previous x
			#pragma omp parallel for
			for (int j = 0; j < N_theta; j++)
			{
				//Initialising
				MatrixXd X(N_x, M);
				MatrixXd X_temp(N_x, M);
				MatrixXd X_prev(N_x, M);
				MatrixXd X_partial(N_x, M_partial);

				VectorXd W_x(N_x);
				VectorXd W_cum(N_x);
				VectorXd U(N_x);
				double U1;
				int index;
				double i_;
				MatrixXd Sigma_prop(M, M);
				MatrixXd noise(N_x, M);

				//load X from file j
				X = X_array[j];
				X_temp = X; //FIX

				//Normalising weights
				W_x = w_x.col(j) / w_x.col(j).sum();
				//Resampling						
				W_cum(0) = W_x(0);
				for (int i = 1; i < N_x; i++)
				{
					W_cum(i) = W_cum(i - 1) + W_x(i);
				}
				U1 = runif_01(rd) / N_x_double;

				index = 0;
				for (int i = 0; i < N_x; i++)
				{
					i_ = i;
					U(i) = U1 + i_ / N_x_double;

					while (U(i) > W_cum(index))
					{
						index++;
					}
				
					//Picking out if index should be replaced
					if (i != index)
					{
						X.row(i) = X_temp.row(index);
						w_x(i, j) = 1.0 / N_x_double;
					}
				}

				X_array[j] = X;

				//current sigma
				for (int jj = 0; jj < M; jj++)
				{
					for (int ii = 0; ii < M; ii++)
					{
						if (ii == jj) {
								Sigma_prop(ii, jj) = delta_t*pow(sigma(j), 2);			
						}
						else {
							Sigma_prop(ii, jj) = 0;
						}
					}
				}

				noise = fun_gen_noise(N_x, M, Mu_prop, Sigma_prop, ind);

				X_prev = X_array[j];

				for (int i = 0; i < N_x; i++)
				{
					F_in_proposal = 0;
					//Proposal
					F_in_proposal = F(j);
					X.row(i) = fun_prop_det(F_in_proposal, M, X_prev.row(i), delta_t) + noise.row(i);
				}
			
				for (int i = 0; i < N_x; i++)
				{
				//weighting
					w_x(i, j) = fun_mvdnorm_ind(trajectory_y.row(k).transpose(), X.row(i).transpose(), sigma_y)(0, 0);			
				}

				//update importance weights
				likelihood_now(j, 0) = w_x.col(j).mean();		
				likelihood_divided(j, 0) = likelihood_acc(j, 0);
				likelihood_divided(j, 1) = likelihood_now(j, 0);
			}

			//tempering
			r = 0;
			phi_r = 0;
			phi_prev = 0;

			while (phi_r < 1)
			{
				//phi_r = cooling(r);
				r++;
				a = phi_prev;
				b = 1;

				//bisection	
				while (b - a > prec)
				{
					c = (a + b) / 2;

					if (fun_ess_r(c, phi_prev, likelihood_now) < alpha_temp*N_theta_double)		//smaller - update upper boundary
					{
						a = a;
						b = c;
					}
					else			//larger - update lower boundary
					{
						a = c;
						b = b;
					}
					phi_r = (a + b) / 2;

					if (phi_r > (1 - prec))			//phi close to one within prec - set to one
					{
						phi_r = 1;
						a = 1;
						b = 1;
					}
				}

				cout << "phi_r" << endl;
				cout << phi_r << endl;

				//compute normalised tempering weights
				W_theta = pow(likelihood_now.array(), (phi_r - phi_prev)) / (pow(likelihood_now.array(), (phi_r - phi_prev))).sum();				//works for likelihood_now as matrix?

				phi_prev = phi_r;

				//compute model evidence
				L_r = (W_theta.transpose() * likelihood_now)(0, 0);

				//compute effective sample size
				ess_theta_r = 1 / pow(W_theta.array(), 2).sum();

				cout << "ess_theta_r" << endl;
				cout << ess_theta_r << endl;

				//compute moments
				mean_F = (W_theta.transpose() * F)(0, 0);
				mean_sigma = (W_theta.transpose() * sigma)(0, 0);
				mean_prop(0, 0) = mean_F;
				mean_prop(1, 0) = mean_sigma;

				para_matrix.col(0) = F;
				para_matrix.col(1) = sigma;

				cout << "weighted mean F" << endl;
				cout << mean_F << endl;
				cout << "weighted mean sigma" << endl;
				cout << mean_sigma << endl;

				cov_prop.fill(0);

				for (int i = 0; i < N_theta; i++)
				{
					cov_prop += W_theta(i, 0) * (para_matrix.row(i) - mean_prop.transpose()).transpose() * (para_matrix.row(i) - mean_prop.transpose());	//fix multiplication
				}

				//resample theta
				F_temp = F;
				sigma_temp = sigma;
				likelihood_now_temp = likelihood_now;
				likelihood_divided_temp = likelihood_divided;
				X_array_temp = X_array;

				//resampling						
				W_theta_cum(0, 0) = W_theta(0, 0);
				for (int j = 1; j < N_theta; j++)
				{
					W_theta_cum(j, 0) = W_theta_cum(j - 1, 0) + W_theta(j, 0);
				}

				U1 = runif_01(rd) / N_theta_double;

				index = 0;
				for (int j = 0; j < N_theta; j++)
				{
					j_ = j;
					U_theta(j) = U1 + j_ / N_theta_double;

					while (U_theta(j) > W_theta_cum(index, 0))
					{
						index++;
					}

					//Picking out if index should be replaced
					if (j != index)
					{
						F(j) = F_temp(index);
						sigma(j) = sigma_temp(index);
						X_array[j] = X_array_temp[index];
						w_x.col(j) = w_x.col(index);
						likelihood_now(j, 0) = likelihood_now_temp(index, 0);
						likelihood_divided(j, 0) = likelihood_divided_temp(index, 0);
						likelihood_divided(j, 1) = likelihood_divided_temp(index, 1);
					}
					//Normalising weights
					W_theta(j, 0) = 1.0 / N_theta_double;
				}

				acc_r = 0;
				acc_theta.fill(0);

				//Rejuvenation
				F_before.push_back(F);
				sigma_before.push_back(sigma);

				cout << "begin rejuvenation" << endl;
				#pragma omp parallel for
				for (int j = 0; j < N_theta; j++)
				{
					//initialising for parallell
					VectorXd F_mcmc(nsteps_1);
					VectorXd sigma_mcmc(nsteps_1);
					MatrixXd prev_mat(2, 1);

					MatrixXd Mu_noise_mat(2, 1);
					Mu_noise_mat(0, 0) = 0;
					Mu_noise_mat(1, 0) = 0;
					Mu_noise_mat.setZero();

					MatrixXd prev_logl_divided(2, 1);

					MatrixXd Mu_mcmc(2, 1);
					Mu_mcmc.fill(0);
					MatrixXd noise_mat(2, 1);
					MatrixXd prop_mat(2, 1);

					MatrixXd Sigma_mcmc(2, 2);

					//MCMC rejuvenation after proposing
					MatrixXd logl_divided_pf(2, 1);
					logl_divided_pf.fill(0);

					double F_prop;
					double sigma_prop;

					//pf
					int T_pf;

					MatrixXd Mu_0_pf(M, 1);
					MatrixXd Sigma_prop_pf(M, M);

					MatrixXd X_pf(N_x, M);
					VectorXd X_prev_pf(M);
					MatrixXd X_temp_pf(N_x, M);

					MatrixXd w_temp_pf(N_x, 1);
					MatrixXd w_pf(N_x, 1);
					MatrixXd W_pf(N_x, 1);
					MatrixXd W_cum_pf(N_x, 1);

					VectorXd likelihood_pf(T);
					VectorXd logl_pf(T);
					logl_pf.fill(0);
					VectorXd ess_pf(T);
					VectorXi k_res_pf(T);
					MatrixXd X_bar_pf(T, M);
					ess_pf.setZero();
					k_res_pf.setZero();
					
					MatrixXd noise_pf(N_x, M);

					//resampling in pf
					int index;
					double U1;
					double i_;
					VectorXd U(N_x);

					//after pf
					double logl;
					double logl_prior;
					double logl_prop;
					double prev_logl;
					double prev_logl_prior;
					double rev_logl_prop;
					double r1;
					double r2;
					double alpha_ratio;

					VectorXd acc_mcmc(nsteps);
					acc_mcmc.fill(0);

					//partial
					MatrixXd X_partial_pf(N_x, M_partial);

					//Initial values
					F_mcmc(0) = F(j);
					sigma_mcmc(0) = sigma(j);

					prev_logl_divided = likelihood_divided.row(j).array().log().transpose();
					prev_logl = prev_logl_divided(0, 0) + prev_logl_divided(1, 0) * phi_r;						//CHECK!!! this is too small

					for (int i = 1; i < nsteps_1; i++)
					{
						prev_mat(0, 0) = F_mcmc(i - 1);
						prev_mat(1, 0) = sigma_mcmc(i - 1);

						noise_mat = fun_gen_noise(1, 2, Mu_noise_mat, cov_prop, dep).transpose();

						prop_mat = mean_prop + rho * (prev_mat - mean_prop) + pow((1 - pow(rho, 2)), 0.5) * noise_mat;

						if (prop_mat(0, 0) < p1 || prop_mat(0, 0) > p2 || prop_mat(1, 0) < 0)
						{
							F_mcmc(i) = F_mcmc(i - 1);
							sigma_mcmc(i) = sigma_mcmc(i - 1);
							acc_append.push_back(0);
							continue;
						}

						//PF for prop_mat
						T_pf = k + 1;

						F_prop = prop_mat(0, 0);
						sigma_prop = prop_mat(1, 0);

						for (int jj = 0; jj < M; jj++)
						{
							for (int ii = 0; ii < M; ii++)
							{
								if (ii == jj) {
									Sigma_prop_pf(ii, jj) = delta_t*pow(sigma_prop, 2);		
								}
								else {
									Sigma_prop_pf(ii, jj) = 0;
								}
							}
						}

						w_pf.fill(0);
						for (int k_pf = 0; k_pf < T_pf; k_pf++)
						{
							//Initialising step
							if (k_pf == 0)
							{
								for (int ii = 0; ii < N_x; ii++)
								{
									for (int jj = 0; jj < M; jj++)
									{
										X_pf(ii, jj) = runif_03(rd);
									}
								}

								for (int i_pf = 0; i_pf < N_x; i_pf++)
								{
									w_pf(i_pf, 0) = 1.0 / N_x_double;
								}
							}

							else
							{
								noise_pf = fun_gen_noise(N_x, M, Mu_prop, Sigma_prop_pf, ind);

								for (int i_pf = 0; i_pf < N_x; i_pf++)
								{
									X_prev_pf = X_pf.row(i_pf);
									//Proposal
									X_pf.row(i_pf) = fun_prop_det(F_prop, M, X_prev_pf, delta_t) + noise_pf.row(i_pf); 
									//weight
									w_pf(i_pf, 0) = fun_mvdnorm_ind(trajectory_y.row(k_pf).transpose(), X_pf.row(i_pf).transpose(), sigma_y)(0, 0);    

									//partially 									
									//for (int iii = 0; iii < M_partial; iii++)
									//{
									//	X_partial_pf.row(i_pf)(iii) = X_pf.row(i_pf)(partial_sequence(iii));
									//}
								
									//w_pf(i_pf, 0) = fun_mvdnorm_ind(trajectory_y_partial.row(k_pf).transpose(), X_partial_pf.row(i_pf).transpose(), sigma_y)(0, 0);     
								}
							}

							W_pf = w_pf / w_pf.sum();

							//Likelihood
							likelihood_pf(k_pf) = w_pf.mean();
							logl_pf(k_pf) = log(w_pf.mean());

							//Ess
							ess_pf(k_pf) = 1 / (W_pf.array()*W_pf.array()).sum();

							if (k_pf != (T_pf - 1))  //no resampling last step
							{
								X_temp_pf = X_pf;
								w_temp_pf = w_pf;
								//Resampling						
								W_cum_pf(0, 0) = W_pf(0, 0);
								for (int ii = 1; ii < N_x; ii++)
								{
									W_cum_pf(ii, 0) = W_cum_pf(ii - 1, 0) + W_pf(ii, 0);
								}
								U1 = runif_01(rd) / N_x_double;

								index = 0;										//write jj or similar other counting
								for (int ii = 0; ii < N_x; ii++)
								{
									i_ = ii;
									U(ii) = U1 + i_ / N_x_double;

									while (U(ii) > W_cum_pf(index, 0))
									{
										index++;
									}
									//Picking out
									if (ii != index)
									{
										X_pf.row(ii) = X_temp_pf.row(index);
									}
									w_pf(ii, 0) = 1.0 / N_x_double;
								}
							} //end resampling in pf
						}

						logl_divided_pf.fill(0);

						for (int ii = 0; ii < k; ii++)
						{
							logl_divided_pf(0, 0) += logl_pf(ii);
						}

						logl_divided_pf(1, 0) = logl_pf(k);
						logl = logl_divided_pf(0, 0) + logl_divided_pf(1, 0) * phi_r;								//CHECK!!! this is big

																													//MCMC step
						prev_mat(0, 0) = F_mcmc(i - 1);
						prev_mat(1, 0) = sigma_mcmc(i - 1);

						Mu_mcmc = mean_prop + rho * (prev_mat - mean_prop);
						Sigma_mcmc = (1 - pow(rho, 2)) * cov_prop;

						logl_prior = fun_logdinvgamma(prop_mat(1, 0), p3, 1.0 / p4);
						logl_prop = log(fun_mvdnorm(prev_mat, Mu_mcmc, Sigma_mcmc)(0, 0));

						prev_logl_prior = fun_logdinvgamma(prev_mat(1, 0), p3, 1.0 / p4);
						rev_logl_prop = log(fun_mvdnorm(prop_mat, Mu_mcmc, Sigma_mcmc)(0, 0)); 
															
						r1 = logl + logl_prior + logl_prop;
						r2 = prev_logl + prev_logl_prior + rev_logl_prop;

						alpha_ratio = exp(r1 - r2);
						
						U1 = runif_01(rd);

						if (U1 < min(alpha_ratio, 1.0))
						{
							F_mcmc(i) = prop_mat(0, 0);
							sigma_mcmc(i) = prop_mat(1, 0);
							prev_logl_divided = logl_divided_pf;
							prev_logl = logl;
							acc_mcmc(i - 1) = 1;
							acc_append.push_back(1);
						}
						else
						{
							F_mcmc(i) = F_mcmc(i - 1);
							sigma_mcmc(i) = sigma_mcmc(i - 1);
							acc_mcmc(i - 1) = 0;
						}
					} //end nsteps in rejuv for theta particle j

					if (acc_mcmc.sum() > 0) //after all steps store likelihood if at least one move has been made
					{
						//theta particles
						F(j) = F_mcmc(nsteps);
						sigma(j) = sigma_mcmc(nsteps);

						//write prev_states to states j 
						X_array[j] = X_pf;
						w_x.col(j) = w_pf.col(0);

						//save likelihood
						likelihood_now(j, 0) = exp(prev_logl_divided(1, 0));
						likelihood_divided(j, 0) = exp(prev_logl_divided(0, 0));
						likelihood_divided(j, 1) = exp(prev_logl_divided(1, 0));
					}

					//accumulate acceptance for rejuvenation at time point r
					acc_r += acc_mcmc.sum();
				} //end rejuvenation for all theta particles at time point r

				cout << "end rejuvenation" << endl;

				F_after.push_back(F);
				sigma_after.push_back(sigma);

				//counting
				T_steps += N_theta_double*nsteps_double;
				T_acc += acc_r;
				acc_ratio_r_path.push_back(T_acc / T_steps);

				cout << "acc ratio" << endl;
				cout << acc_ratio_r_path[acc_ratio_r_path.size() - 1] << endl;

				k_path.push_back(k);
				r_path.push_back(r);
				phi_r_path.push_back(phi_r);

				ess_theta_r_path.push_back(ess_theta_r);
				L_r_path.push_back(L_r);
				count.push_back(count[count.size() - 1] + 1);	

				if (phi_r == 1)
				{
					phi_one.push_back(1);
				}
				else
				{
					phi_one.push_back(0);
				}
			} // end while(phi_r<1)

			for (int ii = 0; ii < N_theta; ii++)
			{
				likelihood_acc(ii, 0) = likelihood_divided(ii, 0) * likelihood_now(ii, 0);
			}
		}//end if k>0 

		F_path.col(k) = F;
		sigma_path.col(k) = sigma;
		k_prev = k;
	}//end for time, reached T_a

	 //Printing
	int t_end = count.size();

	const int Count = 10;      
	string name = "smc2_";    

	ofstream outfstr[Count];      
	for (int i = 0; i < Count; ++i) {   
		outfstr[i].open(name + char('0' + i) + ".txt");
		cout << name + char('0' + i) + ".txt" << endl;
	}

	outfstr[0] << F_path << endl;
	outfstr[1] << sigma_path << endl;

	for (int i = 0; i < t_end; ++i)
	{
		outfstr[2] << acc_ratio_r_path[i] << endl;
		outfstr[3] << L_r_path[i] << endl;
		outfstr[4] << ess_theta_r_path[i] << endl;
		outfstr[5] << phi_r_path[i] << endl;
		outfstr[6] << r_path[i] << endl;
		outfstr[7] << count[i] << endl;
		outfstr[8] << k_path[i] << endl;
		outfstr[9] << phi_one[i] << endl;
	}

	for (int i = 0; i < Count; ++i) {  
		outfstr[i].close();
	}

	// Print MCMC
	const int Count2 = 4;       
	string name2 = "mcmc_";     

	ofstream outfstr2[Count2];      
	for (int i = 0; i < Count2; ++i) {   
		outfstr2[i].open(name2 + char('0' + i) + ".txt");
		cout << name2 + char('0' + i) + ".txt" << endl;
	}

	for (int j = 0; j < N_theta; ++j)
		{
		outfstr2[0] << F_init(j) << ' ';
		outfstr2[1] << F_init(j) << ' ';
		outfstr2[2] << sigma_init(j) << ' ';
		outfstr2[3] << sigma_init(j) << ' ';
		}

	outfstr2[0] << endl;
	outfstr2[1] << endl;
	outfstr2[2] << endl;
	outfstr2[3] << endl;


	for (int i = 0; i < (t_end-1); ++i)
	{
		for (int j = 0; j < N_theta; ++j)
		{
			outfstr2[0] << F_before[i][j] << ' ';
			outfstr2[1] << F_after[i][j] << ' ';
			outfstr2[2] << sigma_before[i][j] << ' ';
			outfstr2[3] << sigma_after[i][j] << ' ';
		}
		outfstr2[0] << endl;
		outfstr2[1] << endl;
		outfstr2[2] << endl;
		outfstr2[3] << endl;
	}	

	for (int i = 0; i < Count2; ++i) {   
		outfstr2[i].close();
	}

	finish = time(0);
	cout << "TOTAL TIME" << endl;
	cout << difftime(finish, start) << endl;

	system("pause");
	return(0);
} //end main
