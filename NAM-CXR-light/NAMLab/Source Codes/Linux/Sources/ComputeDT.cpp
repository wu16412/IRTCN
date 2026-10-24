/*************************************************
author: Bowen Yang (original); NAM-CXR-light adaptation 2026-07-21
describe: compute DAB and DW, and then use them to compute DT.
NAM-CXR-light: DT := DW (bypass DAB×DW chroma gate; DAB≈0 on CXR).
**************************************************/
#include "NAM.h"

void ComputeDAB(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DAB)
{
    for (auto &x:adjacent)
    {
        int row = x[0];
        int col = x[1];
        float D = sqrt(pow(region[row].lab[1] - region[col].lab[1], 2) + pow(region[row].lab[2] - region[col].lab[2], 2));
        DAB.push_back(D);
    }
}

void ComputeDW(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DW)
{
    const int TD = region[0].WLD[0].size();

    for (auto &x : adjacent)
    {
        int row = x[0];
        int col = x[1];
        float D = 0;
        for (int j = 0; j < TD; j++)
        {
            D += pow(region[row].WLD[0][j] - region[col].WLD[0][j], 2);
            D += pow(region[row].WLD[1][j] - region[col].WLD[1][j], 2);
            D += pow(region[row].WLD[2][j] - region[col].WLD[2][j], 2);
        }
        D = sqrt(D);
        DW.push_back(D);
    }
}

void ComputeDT(const list<float> &DAB, const list<float> &DW, list<float> &DT)
{
    // NAM-CXR-light: do not gate texture by chroma DAB.
    (void)DAB;
    for (auto itr = DW.begin(); itr != DW.end(); ++itr)
        DT.push_back(*itr);
}
