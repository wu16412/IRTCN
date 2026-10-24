/*************************************************
author: Bowen Yang (original); NAM-CXR-light adaptation 2026-07-21
describe: compute DC based on intensity (L*) average of each region.
**************************************************/
#include "NAM.h"

void ComputeDC(const vector<Region> &region, const vector<Vec2i> &adjacent, vector<float> &DC)
{
    for (auto &x : adjacent)
    {
        int row = x[0];
        int col = x[1];
        float D = fabs(region[row].lab[0] - region[col].lab[0]);
        DC.push_back(D);
    }
}

void ComputeDC(const vector<Region> &region,const list<Vec2i> &adjacent,list<float> &DC)
{
    for (auto &x:adjacent)
    {
        int row = x[0];
        int col = x[1];
        float D = fabs(region[row].lab[0] - region[col].lab[0]);
        DC.push_back(D);
    }
}
