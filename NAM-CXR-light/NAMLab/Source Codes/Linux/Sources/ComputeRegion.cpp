/*************************************************
author: Bowen Yang
date:2019/04/18
describe: Initialize and update regions.
**************************************************/
#include "NAM.h"
#include "unionFind.h"

void ComputeRegion(const Mat &lab, const vector<vector<int>> &blockMap, vector<Region> &region, int NumSeg, int T, int D)
{
    const int &M = lab.rows;
    const int &N = lab.cols;
    region.resize(NumSeg);
    for (int i = 0; i < NumSeg; i++)
    {
        Region &cur = region[i];
        cur.lab = 0;
        cur.area = 0;
    }
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            const int &label = blockMap[i][j];
            Region &cur = region[label];
            cur.lab += lab.at<Vec3f>(i, j);
            cur.area += 1;
        }
    }
    for (int i = 0; i < NumSeg; i++)
    {
        Region &cur = region[i];
        cur.lab /= cur.area;
        cur.WLD.resize(3);
        for (auto &w : cur.WLD)
            w.resize(T*D);
    }
}

void UpdateRegion(vector<Region> &region, vector<int> &parent, int R1, int R2)
{
    parent[R2] = R1;

    region[R1].lab *= region[R1].area;
    region[R1].lab += region[R2].lab * region[R2].area;
    region[R1].area += region[R2].area;
    region[R1].lab /= region[R1].area;

    const size_t &TD = region[0].WLD[0].size();

    vector<float> sum(3);
    for (int i = 0; i < 3; i++)
    {
        sum[i] = 0;
        for (int j = 0; j < TD; j++)
        {
            sum[i] += region[R1].WLD[i][j];
            sum[i] += region[R2].WLD[i][j];
            region[R1].WLD[i][j] += region[R2].WLD[i][j];
        }
        if (!(sum[i] < 1e-8f))
            for (int j = 0; j < TD; j++)
                region[R1].WLD[i][j] /= sum[i];
    }
}

void UpdateRegion(vector<Region> &region, vector<int> &LabelC, int NumSeg, int T, int D)
{
    vector<Region>temp;
    for (int i = 0; i < NumSeg; i++)
    {
        Region r;
        r.lab = 0;
        r.area = 0;
        temp.push_back(r);
    }
    for (auto &cur : temp)
    {
        cur.WLD.resize(3);
        for (auto &w : cur.WLD)
        {
            w.resize(T*D);
            for (auto &x : w)
                x = 0;
        }
    }
    for (int i = 0; i < region.size(); i++)
    {
        temp[LabelC[i]].lab += region[i].lab*region[i].area;
        temp[LabelC[i]].area += region[i].area;
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < T*D; j++)
                temp[LabelC[i]].WLD[k][j] += region[i].WLD[k][j];
    }
    for (auto &cur : temp)
    {
        cur.lab /= cur.area;
        vector<float> sum(3);
        for (int i = 0; i < 3; i++)
        {
            sum[i] = 0;
            for (auto &x : cur.WLD[i])
                sum[i] += x;
            if (!(sum[i] < 1e-8f))
                for (auto &x : cur.WLD[i])
                    x /= sum[i];
        }
    }
    region = std::move(temp);
}
