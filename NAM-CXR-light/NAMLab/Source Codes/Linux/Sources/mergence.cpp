/*************************************************
author: Bowen Yang
date:2019/10/22
describe: Merge blocks after image representation.
**************************************************/

#include "NAM.h"
#include "unionFind.h"

void mergence(unordered_map<string, double> &param,vector<Coordinate> &P, vector<ColorVariance> &R, int &NumSeg,vector<vector<int>> &blockMap)
{
    clock_t s, e;
    s = clock();

    const double &U = param["u"];
    const double &Var = param["var"];

    const int &M = blockMap.size();
    const int &N = blockMap[0].size();

    vector<int> parent;
    for (int i = 0; i < P.size(); i++) parent.push_back(i);

    for (int k = 0; k < P.size(); k++)
    {
        int i, j;
        //West
        j = P[k].SP[1] - 1;
        if (j >= 0)
        {
            for (i = P[k].EP[0]; i >= 0 && i >= P[k].SP[0]; i--)
            {
                const int &No = blockMap[i][j];
                int root1 = Find_Parent(parent, k);
                int root2 = Find_Parent(parent, No);
                if (root1 == root2)continue;

                const Vec3d &c1 = R[root1].color;
                const Vec3d &c2 = R[root2].color;

                float L = abs(c1[0] - c2[0]);
                float A = abs(c1[1] - c2[1]);
                float B = abs(c1[2] - c2[2]);

                Vec3d updatedVariance;
                for (int i = 0; i < 3; i++)
                    updatedVariance[i] = (R[root1].variance[i] * P[root1].area + R[root2].variance[i] * P[root2].area) / (float)(P[root1].area + P[root2].area)
                                         + P[root1].area*P[root2].area*pow(R[root1].variance[i] - R[root2].variance[i], 2) / pow(P[root1].area + P[root2].area, 2);

                // NAM-CXR-light: merge gate on L* / var_L only.
                (void)A; (void)B;
                if (L > U || updatedVariance[0] > Var)continue;

                if (root1 > root2)swap(root1, root2);

                Join(parent, root1, root2);
                for (int i = 0; i < 3; i++)
                    R[root1].color[i] = (R[root1].color[i] * P[root1].area + R[root2].color[i] * P[root2].area) / (P[root1].area + P[root2].area);
                for (int i = 0; i < 3; i++)
                    R[root1].variance[i] = (R[root1].variance[i] * P[root1].area + R[root2].variance[i] * P[root2].area) / (float)(P[root1].area + P[root2].area)
                                           + P[root1].area*P[root2].area*pow(R[root1].color[i] - R[root2].color[i], 2) / pow(P[root1].area + P[root2].area, 2);
                P[root1].area += P[root2].area;
            }
        }
        //North
        i = P[k].SP[0] - 1;
        if (i >= 0)
        {
            for (j = P[k].SP[1]; j <= N && j <= P[k].EP[1]; j++)
            {
                const int &No = blockMap[i][j];
                int root1 = Find_Parent(parent, k);
                int root2 = Find_Parent(parent, No);
                if (root1 == root2)continue;

                const Vec3d &c1 = R[root1].color;
                const Vec3d &c2 = R[root2].color;

                float L = abs(c1[0] - c2[0]);
                float A = abs(c1[1] - c2[1]);
                float B = abs(c1[2] - c2[2]);

                Vec3d updatedVariance;
                for (int i = 0; i < 3; i++)
                    updatedVariance[i] = (R[root1].variance[i] * P[root1].area + R[root2].variance[i] * P[root2].area) / (float)(P[root1].area + P[root2].area)
                                         + P[root1].area*P[root2].area*pow(R[root1].variance[i] - R[root2].variance[i], 2) / pow(P[root1].area + P[root2].area, 2);

                // NAM-CXR-light: merge gate on L* / var_L only.
                (void)A; (void)B;
                if (L > U || updatedVariance[0] > Var)continue;

                if (root1 > root2)swap(root1, root2);

                Join(parent, root1, root2);
                for (int i = 0; i < 3; i++)
                    R[root1].color[i] = (R[root1].color[i] * P[root1].area + R[root2].color[i] * P[root2].area) / (P[root1].area + P[root2].area);
                for (int i = 0; i < 3; i++)
                    R[root1].variance[i] = (R[root1].variance[i] * P[root1].area + R[root2].variance[i] * P[root2].area) / (float)(P[root1].area + P[root2].area)
                                           + P[root1].area*P[root2].area*pow(R[root1].color[i] - R[root2].color[i], 2) / pow(P[root1].area + P[root2].area, 2);
                P[root1].area += P[root2].area;
            }
        }
    }

    unordered_map<int, int> Label;
    for (int i = 0; i < parent.size(); i++)
        if (Find_Parent(parent, i) == i)
            Label[i] = NumSeg++;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            blockMap[i][j] = Label[Find_Parent(parent, blockMap[i][j])];

    e = clock();
    param["mergence"] = (e - s) / (float)CLOCKS_PER_SEC;
}
