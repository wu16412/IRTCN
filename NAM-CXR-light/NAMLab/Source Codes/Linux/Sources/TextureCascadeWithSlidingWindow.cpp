/*************************************************
author: Bowen Yang
date:2019/04/19
describe: compute cascade F.
**************************************************/
#include "NAM.h"

void TextureCascadeWithSlidingWindow(const vector<vector<int>> &F, int M, int N, int R, int K, vector<vector<int>> &CascadeF)
{
	int countConv;
	for (int i = R; i < M - R; i++) {
		for (int j = R; j < N - R; j++) {
			countConv = 0;
			for (int u = j - R; u <= j + R; u++) {
				for (int v = i - R; v <= i + R; v++) {
					CascadeF[i*N + j][countConv] = F[v][u];
					countConv = countConv + 1;
				}
			}
		}
	}

	//fill
	for (int i = R; i < M - R; i++) {
		int p1 = R, p2 = N - 1 - R;
		for (int k = 1; k <= R; k++)
		{
			for (int m = 0; m < K*K; m++)
			{
				CascadeF[i*N + p1 - k][m] = CascadeF[i*N + p1][m];
				CascadeF[i*N + p2 + k][m] = CascadeF[i*N + p2][m];
			}
		}
	}
	for (int j = R; j < N - R; j++) {
		int p1 = R, p2 = M - 1 - R;
		for (int k = 1; k <= R; k++)
		{
			for (int m = 0; m < K*K; m++)
			{
				CascadeF[(p1 - k)*N + j][m] = CascadeF[p1*N + j][m];
				CascadeF[(p2 + k)*N + j][m] = CascadeF[p2*N + j][m];
			}
		}
	}
	for (int m = 1; m <= R; m++) {
		int i1 = R - m;
		int i2 = M - 1 - R + m;
		int p1 = R, p2 = N - 1 - R;
		for (int k = 1; k <= R; k++)
		{
			for (int m = 0; m < K*K; m++)
			{
				CascadeF[i1*N + p1 - k][m] = CascadeF[i1*N + p1][m];
				CascadeF[i1*N + p2 + k][m] = CascadeF[i1*N + p2][m];

				CascadeF[i2*N + p1 - k][m] = CascadeF[i2*N + p1][m];
				CascadeF[i2*N + p2 + k][m] = CascadeF[i2*N + p2][m];
			}
		}
	}
}
