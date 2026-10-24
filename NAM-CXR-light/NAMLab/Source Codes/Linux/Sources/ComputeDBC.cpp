/*************************************************
author: Bowen Yang (original); NAM-CXR-light adaptation 2026-07-21
describe: Compute Sum and Count, and then use them to compute DBC (L*-only).
**************************************************/
#include "NAM.h"

void ComputeSumAndCount(const Mat &lab, const vector<vector<int>> &blockMap, int M, int N, int NumSeg, list<float> &Sum, list<float> &Count, const list<Vec2i> &adjacent)
{
    unordered_map<long long, float> summatrix;
    unordered_map<long long, int> countmatrix;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            int minX = max(0, i - 1);
            int maxX = min(i + 1, M - 1);
            int minY = max(0, j - 1);
            int maxY = min(j + 1, N - 1);
            for (int ii = minX; ii <= maxX; ii++)
                for (int jj = minY; jj <= maxY; jj++)
                    if (blockMap[i][j] != blockMap[ii][jj])
                    {
                        long long pos = (long long)blockMap[i][j] * (long long)NumSeg + (long long)blockMap[ii][jj];
                        float dL = fabs(lab.at<Vec3f>(i, j)[0] - lab.at<Vec3f>(ii, jj)[0]);
                        if (summatrix.find(pos) == summatrix.end())
                            summatrix[pos] = dL;
                        else
                            summatrix[pos] += dL;
                        if (countmatrix.find(pos) == countmatrix.end())
                            countmatrix[pos] = 1;
                        else
                            countmatrix[pos] += 1;
                    }
        }
    for (auto &x : adjacent)
    {
        int row = x[0];
        int col = x[1];
        long long pos1 = (long long)row * (long long)NumSeg + (long long)col;
        long long pos2 = (long long)col * (long long)NumSeg + (long long)row;
        Count.push_back(max(countmatrix[pos1], countmatrix[pos2]));
        Sum.push_back(max(summatrix[pos1], summatrix[pos2]));
    }
}

void ComputeDBC(const list<float> &Sum, const list<float> &Count, list<float> &DBC)
{
    list<float>::const_iterator itrsum = Sum.begin();
    list<float>::const_iterator itrcount = Count.begin();

    for (; itrsum != Sum.end(); itrsum++, itrcount++)
        if ((*itrcount) > 0)
            DBC.push_back((*itrsum) / (*itrcount));
}
