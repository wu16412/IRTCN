/*************************************************
author: Bowen Yang (original); NAM-CXR-light adaptation 2026-07-21
describe: NAM based hierarchical image segmentation (grayscale CXR light).
**************************************************/
#include "NAM.h"
#include "rgb2lab.h"

// Set by NAM(); used by scanning() to dump area-weighted WLD bins.
string g_NAMFeatOutPath;

void NAM(unordered_map<string, double> &param, const string &imgPath, vector<Mat> &map, vector<Mat> &result)
{
	Mat img = imread(imgPath, IMREAD_UNCHANGED);
	if (!img.data)
		return;
	// Accept gray CXR; convert to 3-channel for the rest of the pipeline.
	if (img.channels() == 1)
		cvtColor(img, img, COLOR_GRAY2BGR);
	else if (img.channels() == 4)
		cvtColor(img, img, COLOR_BGRA2BGR);
	else if (img.channels() != 3)
		return;

	const int &M = img.rows;
	const int &N = img.cols;

	Mat lab = Mat::zeros(M, N, CV_32FC3);
	vector<Coordinate> P;
    vector<ColorVariance> R;
    int NumSeg = 0;
    vector<vector<int>> blockMap(M);
    for (auto &x : blockMap) x.resize(N);
    vector<Region> region;
    vector<Vec2i> adjacent;
    list<Vec2i> adjacentList;

	switch ((int)param["lab"])
	{
	case 0://D50
		lab.convertTo(lab, CV_32FC3);
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < N; j++)
			{
				Vec3b &c = img.at<Vec3b>(i, j);
				Vec3f &l = lab.at<Vec3f>(i, j);
				float L, A, B;
				RGBToLabD50(c[2], c[1], c[0], L, A, B);
				l[0] = L;
				// NAM-CXR-light: drop chroma (CXR has no meaningful a*/b*).
				l[1] = 0.f;
				l[2] = 0.f;
			}
		}
		break;
	case 1://D65
		img.convertTo(img, CV_32FC3, 1.0 / 255, 0);
		cvtColor(img, lab, COLOR_BGR2Lab);
		img.convertTo(img, CV_8UC3, 255.0, 0);
		for (int i = 0; i < M; i++)
			for (int j = 0; j < N; j++)
			{
				Vec3f &l = lab.at<Vec3f>(i, j);
				l[1] = 0.f;
				l[2] = 0.f;
			}
		break;
	}

	// Feature export: <imgDir>/Output/<stem>_wld144.txt
	// Caller may pre-set g_NAMFeatOutPath (e.g. run_dataset → outDir).
	if (g_NAMFeatOutPath.empty())
	{
		string x = imgPath;
		size_t slash = x.find_last_of("/\\");
		string dir = (slash == string::npos) ? string("./") : x.substr(0, slash + 1);
		string base = (slash == string::npos) ? x : x.substr(slash + 1);
		size_t dot = base.find_last_of('.');
		if (dot != string::npos) base = base.substr(0, dot);
		g_NAMFeatOutPath = dir + "Output/" + base + "_wld144.txt";
	}

	representation(param,lab,P,R,blockMap);

	mergence(param,P,R,NumSeg,blockMap);

	removeRemnant(param,lab,NumSeg,blockMap,region,map,result,adjacent,adjacentList);

	scanning(param,region,blockMap,adjacentList,lab,img,map,result);
}
