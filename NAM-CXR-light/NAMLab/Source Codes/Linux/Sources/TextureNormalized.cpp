/*************************************************
author: Bowen Yang
date:2019/04/19
describe: normalize W.
**************************************************/
#include "NAM.h"

void TextureNormalized(vector<vector<float>> &W, int NumSeg, int T, int D)
{
	vector<float> sum(NumSeg);
	for (int i = 0; i < NumSeg; i++) {
		sum[i] = 0;
		for (int j = 0; j < T*D; j++)
			sum[i] += W[i][j];
	}
	for (int i = 0; i < NumSeg; i++) {
		for (int j = 0; j < T*D; j++)
			if (sum[i] < 1e-8f)
				continue;
			else
				W[i][j] /= sum[i];
	}
}
