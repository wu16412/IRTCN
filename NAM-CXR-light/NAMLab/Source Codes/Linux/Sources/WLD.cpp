/*************************************************
author: Bowen Yang
date:2019/04/19
describe: calculate the index of each pixel in the [T * D] array (compute F).
**************************************************/

#include "NAM.h"

void WLD(const Mat &gray, vector<vector<int>> &F, int M, int N, int R, const vector<int> &f, int T, int D)
{
	vector<vector<float>> Excitation(M);
	vector<vector<float>> Orientation(M);
	vector<vector<int>> Mask1(M);
	vector<vector<int>> Mask2(M);

	for (int i = 0; i < M; i++)
	{
		Excitation[i].resize(N);
		Orientation[i].resize(N);
		Mask1[i].resize(N);
		Mask2[i].resize(N);
	}


	float BELTA = 5; // to avoid that center pixture is equal to zero
	float ALPHA = 3; // like a lens to magnify or shrink the difference between neighbors
	float EPSILON = 1e-7f;
	float PI = 3.141592653589f;

	float v00, v01, v10, v11;
	float N1, N3, N5, N7;
	int countConv;
	for (int i = R; i < M - R; i++) {
		for (int j = R; j < N - R; j++) {
			// step 1 compute differential excitationt
			v01 = gray.at<uchar>(i, j) + BELTA;
			countConv = 0;
			v00 = 0;
			for (int u = i - R; u <= i + R; u++) {
				for (int v = j - R; v <= j + R; v++) {
					v00 += gray.at<uchar>(u, v) * f[countConv];
					countConv = countConv + 1;
				}
			}
			if (!(abs(v01) <= 1e-7f))
				Excitation[i][j] = atan(ALPHA*(v00 / v01));
			else
				Excitation[i][j] = 0.1f;

			// step 2 compute gradient orientation
			N1 = gray.at<uchar>(i - R, j);	//up
			N5 = gray.at<uchar>(i + R, j);	//down
			N7 = gray.at<uchar>(i, j - R);	//left
			N3 = gray.at<uchar>(i, j + R);	//right
			if (abs(N7 - N3) < EPSILON)
				Orientation[i][j] = 0;
			else
			{
				v10 = N5 - N1; v11 = N7 - N3;
				Orientation[i][j] = atan(v10 / v11);
				Orientation[i][j] = Orientation[i][j] * 180.f / PI;
				if (v11 > EPSILON && v10 > EPSILON)
					Orientation[i][j] = Orientation[i][j] + 0.f;
				else if (v11 < -EPSILON && v10 >  EPSILON)
					Orientation[i][j] = Orientation[i][j] + 180.f;
				else if (v11 < -EPSILON && v10 < -EPSILON)
					Orientation[i][j] = Orientation[i][j] + 180.f;
				else if (v11 > EPSILON && v10 < -EPSILON)
					Orientation[i][j] = Orientation[i][j] + 360.f;
			}
		}
	}
	//fill
	/*
			0 0 0 0 0 0				0 0 0 0 0 0
			0 0 0 0 0 0				0 0 0 0 0 0
			0 0 1 2 0 0				1 1 1 2 2 2
			0 0 3 4 0 0		->		3 3 3 4 4 4
			0 0 0 0 0 0				0 0 0 0 0 0
			0 0 0 0 0 0				0 0 0 0 0 0
	*/
	for (int i = R; i < M - R; i++) {
		int p1 = R, p2 = N - 1 - R;
		for (int k = 1; k <= R; k++)
		{
			Excitation[i][p1 - k] = Excitation[i][p1];
			Excitation[i][p2 + k] = Excitation[i][p2];
			Orientation[i][p1 - k] = Orientation[i][p1];
			Orientation[i][p2 + k] = Orientation[i][p2];
		}
	}
	/*
			0 0 0 0 0 0				0 0 1 2 0 0
			0 0 0 0 0 0				0 0 1 2 0 0
			1 1 1 2 2 2				1 1 1 2 2 2
			3 3 3 4 4 4		->		3 3 3 4 4 4
			0 0 0 0 0 0				0 0 3 4 0 0
			0 0 0 0 0 0				0 0 3 4 0 0
	*/
	for (int j = R; j < N - R; j++) {
		int p1 = R, p2 = M - 1 - R;
		for (int k = 1; k <= R; k++)
		{
			Excitation[p1 - k][j] = Excitation[p1][j];
			Excitation[p2 + k][j] = Excitation[p2][j];;
			Orientation[p1 - k][j] = Orientation[p1][j];
			Orientation[p2 + k][j] = Orientation[p2][j];
		}
	}
	/*
			0 0 1 2 0 0				1 1 1 2 2 2
			0 0 1 2 0 0				1 1 1 2 2 2
			1 1 1 2 2 2				1 1 1 2 2 2
			3 3 3 4 4 4		->		3 3 3 4 4 4
			0 0 3 4 0 0				3 3 3 4 4 4
			0 0 3 4 0 0				3 3 3 4 4 4
	*/
	for (int m = 1; m <= R; m++) {
		int i1 = R - m;
		int i2 = M - 1 - R + m;
		int p1 = R, p2 = N - 1 - R;
		for (int k = 1; k <= R; k++) {
			Excitation[i1][p1 - k] = Excitation[i1][p1];
			Excitation[i1][p2 + k] = Excitation[i1][p2];
			Orientation[i1][p1 - k] = Orientation[i1][p1];
			Orientation[i1][p2 + k] = Orientation[i1][p2];

			Excitation[i2][p1 - k] = Excitation[i2][p1];
			Excitation[i2][p2 + k] = Excitation[i2][p2];
			Orientation[i2][p1 - k] = Orientation[i2][p1];
			Orientation[i2][p2 + k] = Orientation[i2][p2];
		}
	}

	float stepT = PI / (float)T;
	float stepD = 360 / (float)D;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			float startT = -PI / (float)2;
			for (int m = 0; m < T; m++) {
				if (Excitation[i][j] >= (startT + m * stepT) && Excitation[i][j] < (startT + (m + 1) * stepT))
				{
					Mask1[i][j] = m;
					break;
				}
			}
			for (int m = 0; m < D; m++) {
				if (Orientation[i][j] >= (m * stepD) && Orientation[i][j] < (m + 1) * stepD)
				{
					Mask2[i][j] = m;
					break;
				}
			}
		}
	}
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			F[i][j] = Mask1[i][j] * D + Mask2[i][j];
}
