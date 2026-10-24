/*************************************************
author: Bowen Yang
date:2019/04/18
describe: compute SI.
**************************************************/

#include "NAM.h"

void ComputeSI(const vector<vector<int>> &blockMap, int M, int N, int NumSeg, list<float> &SI, const int Windows, const list<Vec2i> &adjacent)
{
    int K = 2 * Windows + 1;
    vector<vector<int>> LabelCascade(M*N);
    vector<int> LabelMode(M*N);
    unordered_map<long long, int> SImatrix;

    for (auto &l : LabelCascade)
    {
        l.resize(K*K);
        for (auto &x : l)
            x = 0;
    }

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            if (i < Windows || i > M - 1 - Windows || j < Windows || j > N - 1 - Windows)
                continue;
            else
            {
                int count = 0;
                for (int ii = i - Windows; ii <= i + Windows; ii++)
                    for (int jj = j - Windows; jj <= j + Windows; jj++)
                        LabelCascade[i*N + j][count++] = blockMap[ii][jj];
            }
        }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            if (i < Windows || i > M - 1 - Windows || j < Windows || j > N - 1 - Windows)
                LabelMode[i*N + j] = blockMap[i][j];
            else
            {
                sort(LabelCascade[i*N + j].begin(), LabelCascade[i*N + j].end(), greater<int>());
                if (LabelCascade[i*N + j][0] == LabelCascade[i*N + j][K*K - 2])
                    LabelMode[i*N + j] = LabelCascade[i*N + j][0];
                else
                {
                    int temp = LabelCascade[i*N + j][0];
                    int max = 1;
                    int cur = 1;
                    int m;
                    for (m = 1; m < K*K; m++)
                    {
                        if (LabelCascade[i*N + j][m] != LabelCascade[i*N + j][m - 1])
                        {
                            if (cur > max)
                            {
                                max = cur;
                                temp = LabelCascade[i*N + j][m - 1];
                            }
                            cur = 1;
                        }
                        else
                            cur++;
                    }
                    if (cur > max)
                    {
                        max = cur;
                        temp = temp = LabelCascade[i*N + j][m - 1];
                    }
                    LabelMode[i*N + j] = temp;
                }
            }
        }

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            long long pos = (long long)blockMap[i][j] * (long long)NumSeg + (long long)LabelMode[i*N + j];
            if (blockMap[i][j] != LabelMode[i*N + j])
                if (SImatrix.find(pos) == SImatrix.end())
                    SImatrix[pos] = 1;
                else
                    SImatrix[pos] += 1;
        }

    for (auto &x : adjacent)
    {
        int row = x[0];
        int col = x[1];
        long long pos1 = (long long)row * (long long)NumSeg + (long long)col;
        long long pos2 = (long long)col * (long long)NumSeg + (long long)row;
        SI.push_back(min(SImatrix[pos1], SImatrix[pos2]));
    }
}
