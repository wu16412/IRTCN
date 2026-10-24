/*************************************************
author: Bowen Yang
date:2019/04/19
describe: compute [T*D] array for each region based on cascade F.
**************************************************/
#include "NAM.h"

void AccumulateRangeofTexture(const vector<vector<int>> &blockMap, const vector<vector<int>> &CascadeF, int M, int N, int K, vector<vector<float>> &W)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            int seg = blockMap[i][j];
            for (int m = 0; m < K*K; m++)
            {
                W[seg][CascadeF[i*N + j][m]] += 1;
            }
        }
}
