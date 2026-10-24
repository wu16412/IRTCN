/*************************************************
author: Bowen Yang
date:2019/04/18
describe: compute adjacency list based on blockmap.The adjacency list records which two areas are adjacent.
**************************************************/
#include "NAM.h"

void ComputeNeighbor(const vector<vector<int>> &blockMap, int M, int N, int NumSeg, vector<Vec2i> &adjacent)
{
    map<long long, int> neighbormatrix;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int minX = max(0, i - 1);
            int maxX = min(i + 1, M - 1);
            int minY = max(0, j - 1);
            int maxY = min(j + 1, N - 1);
            for (int ii = minX; ii <= maxX; ii++)
            {
                for (int jj = minY; jj <= maxY; jj++)
                {
                    if (blockMap[i][j] != blockMap[ii][jj] && neighbormatrix.find((long long)blockMap[i][j] * (long long)NumSeg + (long long)blockMap[ii][jj]) == neighbormatrix.end())
                        neighbormatrix[(long long) blockMap[i][j] * (long long) NumSeg + (long long) blockMap[ii][jj]] = 1;
                }
            }
        }
    }
    for (auto&x : neighbormatrix)
        adjacent.push_back(Vec2i(x.first / (long long) NumSeg, x.first % (long long) NumSeg));
}

void UpdateNeighbor(vector<Vec2i> &adjacent, vector<int> &LabelC, int NumSeg)
{
    map<long long, int> neighbormatrix;

    for (int i = 0; i < adjacent.size(); i++)
    {
        int row = LabelC[adjacent[i][0]];
        int col = LabelC[adjacent[i][1]];
        if (row != col && neighbormatrix.find((long long)row * (long long)NumSeg + (long long)col) == neighbormatrix.end())
            neighbormatrix[(long long)row * (long long)NumSeg + (long long)col] = 1;
    }
    adjacent.clear();
    for (auto&x : neighbormatrix)
        adjacent.push_back(Vec2i(x.first / (long long) NumSeg, x.first % (long long) NumSeg));
}
