/*************************************************
author: Bowen Yang
date:2019/10/22
describe: header
**************************************************/
#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <list>
#include <cmath>
#include <functional>
#include <ctime>
#include <string>
#include <cstdio>

using namespace std;
using namespace cv;

typedef struct
{
	Vec2i SP;
	Vec2i EP;
    float area;
}Coordinate;

typedef struct
{
    Vec3d color;
    Vec3d variance;
}ColorVariance;

typedef struct
{
	Vec3f lab;
	float area;
	vector<vector<float>> WLD;
}Region;


// NAM-CXR-light: set by NAM() before scanning(); empty disables export.
extern string g_NAMFeatOutPath;

void NAM(unordered_map<string, double> &param, const string &imgPath, vector<Mat> &map, vector<Mat> &result);

void representation(unordered_map<string, double> &param,const Mat &lab, vector<Coordinate> &P, vector<ColorVariance> &R,vector<vector<int>> &blockMap);

void mergence(unordered_map<string, double> &param,vector<Coordinate> &P, vector<ColorVariance> &R, int &NumSeg,vector<vector<int>> &blockMap);

void removeRemnant(unordered_map<string, double> &param,const Mat &lab,int &NumSeg,vector<vector<int>> &blockMap,vector<Region> &region,vector<Mat> &map, vector<Mat> &result,vector<Vec2i> &adjacent,list<Vec2i> &adjacentList);

void scanning(unordered_map<string, double> &param, vector<Region> &region, vector<vector<int>> &blockMap, list<Vec2i> &adjacent,const Mat &lab, const Mat &img, vector<Mat> &map, vector<Mat> &result);

void WLD(const Mat &gray, vector<vector<int>> &F, int M, int N, int R, const vector<int> &f, int T, int D);

void TextureCascadeWithSlidingWindow(const vector<vector<int>> &F, int M, int N, int R, int K, vector<vector<int>> &CascadeF);

void AccumulateRangeofTexture(const vector<vector<int>> &blockMap, const vector<vector<int>> &CascadeF, int M, int N, int K, vector<vector<float>> &W);

void TextureNormalized(vector<vector<float>> &W, int NumSeg, int T, int D);

void ComputeNeighbor(const vector<vector<int>> &blockMap, int M, int N, int NumSeg, vector<Vec2i> &adjacent);

void ComputeSumAndCount(const Mat &lab, const vector<vector<int>> &blockMap, int M, int N, int NumSeg, list<float> &Sum, list<float> &Count, const list<Vec2i> &adjacent);

void ComputeDBC(const list<float> &Sum, const list<float> &Count, list<float> &DBC);

void ComputeRegion(const Mat &lab, const vector<vector<int>> &blockMap, vector<Region> &region, int NumSeg, int T, int D);

void ComputeSI(const vector<vector<int>> &blockMap, int M, int N, int NumSeg, list<float> &SI, const int Windows, const list<Vec2i> &adjacent);

void ComputeDC(const vector<Region> &region, const vector<Vec2i> &adjacent, vector<float> &DC);

void ComputeDC(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DC);

void ComputeDN(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DN, float t, float ThNc);

void ComputeDAB(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DAB);

void ComputeDW(const vector<Region> &region, const list<Vec2i> &adjacent, list<float> &DW);

void ComputeDT(const list<float> &DAB, const list<float> &DW, list<float> &DT);

void ComputeDist(unordered_map<string, double> &param,const list<float> &DC, const list<float> &DBC, const list<float> &DT, const list<float> &SI, const list<float> &DN, const float &SumDC, const float &SumDBC, const float &SumDT, const float &alpha, const float &beta, const float &gamma, list<float> &Dist);

void ComputeDist(unordered_map<string, double> &param,list<float>::iterator &dcItr, list<float>::iterator &dbcItr, list<float>::iterator &dtItr, list<float>::iterator &siItr, list<float>::iterator &dnItr, const float &SumDC, const float &SumDBC, const float &SumDT, const float &alpha, const float &beta, const float &gamma, float &d);

void ComputeThreshold(const list<float> &Dist, const float &Percent, float &THRESHOLD);

void UpdateRegion(vector<Region> &region, vector<int> &parent, int R1, int R2);

void UpdateRegion(vector<Region> &region, vector<int> &LabelC, int NumSeg, int T, int D);

void UpdateProperty(list<Vec2i> &adjacent, list<float> &DC, list<float> &DBC, list<float> &DT, list<float> &SI, list<float> &DN, list<float> &Sum, list<float> &Count, list<float> &DAB, list<float> &DW, float& SumDC, float &SumDT, float &SumDBC, const vector<Region> &region, vector<int> &parent, const int &R1, const int &R2, const float &t, const float &ThNc);

void UpdateNeighbor(vector<Vec2i> &adjacent, vector<int> &LabelC, int NumSeg);
