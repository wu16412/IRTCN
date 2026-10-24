/*************************************************
author: Bowen Yang
date:2019/04/19
describe: compute DN based on the area of each region.
**************************************************/
#include "NAM.h"

void ComputeDN(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DN, float t, float ThNc)
{
    for (auto &x : adjacent)
    {
        int row = x[0];
        int col = x[1];
        float N1 = pow(min(region[row].area, ThNc), 1 / t);
        float N2 = pow(min(region[col].area, ThNc), 1 / t);
        float N = (N1*N2) / (N1 + N2);
        DN.push_back(N);
    }
}
