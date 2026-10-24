/*************************************************
author: Bowen Yang
date:2019/05/21
describe: Remove small remnant regions.
**************************************************/

#include "NAM.h"
#include "unionFind.h"

static void MergeSmallRegion(const vector<Region> &region, const vector<float> &DC, const vector<Vec2i> &adjacent, vector<int> &LabelC, const int &NumSeg, const float &SIZE);

void removeRemnant(unordered_map<string, double> &param,const Mat &lab,int &NumSeg,vector<vector<int>> &blockMap,vector<Region> &region,vector<Mat> &map, vector<Mat> &result,vector<Vec2i> &adjacent,list<Vec2i> &adjacentList)
{
    clock_t s, e;
    s = clock();

    const int &M = blockMap.size();
    const int &N = blockMap[0].size();
    const int T = 6;
    const int D = 8;
    float SIZE = M * N / param["size"];     //size threshold

    ComputeRegion(lab,blockMap,region,NumSeg,T,D);
    ComputeNeighbor(blockMap, M, N, NumSeg, adjacent);

    int flag = 0;
    for (auto &x : region)
        if (x.area < SIZE)
            flag = 1;
    while (flag)
    {
        if (NumSeg <= param["saveHir"])
        {
            //Output label matrix
            Mat m = Mat::zeros(M, N, CV_32FC1);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    m.at<float>(i, j) = blockMap[i][j] + 1;
            map.push_back(std::move(m));
            m.release();

            if (param["mode"] == 1) {
                //Output color image
                Mat r = Mat::zeros(M, N, CV_32FC3);
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        r.at<Vec3f>(i, j) = region[blockMap[i][j]].lab;
                cvtColor(r, r, COLOR_Lab2BGR);
                r.convertTo(r, CV_8UC3, 255.0, 0);
                result.push_back(std::move(r));
                r.release();
            }
        }
        flag = 0;
        vector<int> LabelC(NumSeg);
        vector<float> DC;

        ComputeDC(region, adjacent, DC);
        MergeSmallRegion(region, DC, adjacent, LabelC, NumSeg, SIZE);
        vector<int>::iterator maxpos = max_element(LabelC.begin(), LabelC.end());
        NumSeg = *maxpos + 1;

        UpdateNeighbor(adjacent, LabelC, NumSeg);
        UpdateRegion(region, LabelC, NumSeg, T, D);

        //update blockMap
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                blockMap[i][j] = LabelC[blockMap[i][j]];

        for (auto &x : region)
            if (x.area < SIZE)
            {
                flag = 1;
                break;
            }
    }

    for(auto &x:adjacent)
        adjacentList.push_back(x);

    param["seg"] = NumSeg;
    e = clock();
    param["remnant"] = (e - s) / (float)CLOCKS_PER_SEC;
}

static void MergeSmallRegion(const vector<Region> &region, const vector<float> &DC, const vector<Vec2i> &adjacent, vector<int> &LabelC, const int &NumSeg, const float &SIZE)
{
	vector<int> ClosestIndex;
	vector<float> ClosestValue;
	ClosestIndex.resize(NumSeg);
	ClosestValue.resize(NumSeg);
	for (int i = 0; i < NumSeg; i++)
	{
		ClosestIndex[i] = i;
		ClosestValue[i] = 9999;
	}

	for (int i = 0; i < adjacent.size(); i++)
	{
		if (region[adjacent[i][0]].area <= SIZE && ClosestValue[adjacent[i][0]] > DC[i])
		{
			ClosestIndex[adjacent[i][0]] = adjacent[i][1];
			ClosestValue[adjacent[i][0]] = DC[i];
		}
	}

	for (int i = 0; i < NumSeg; i++)
	{
		if (region[i].area <= SIZE)
		{
			int parent = Find_Parent(ClosestIndex, i);
			ClosestIndex[i] = parent;
		}
	}

	unordered_map<int, int> Label;
	int index = 0;
	for (int i = 0; i < ClosestIndex.size(); i++)
	{
		if (Label.find(ClosestIndex[i]) == Label.end())
			Label[ClosestIndex[i]] = index++;
	}
	for (int i = 0; i < NumSeg; i++)
		LabelC[i] = Label[ClosestIndex[i]];
}

