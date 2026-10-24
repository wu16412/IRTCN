/*************************************************
author: Bowen Yang
date:2019/04/18
describe: 
**************************************************/

#include "NAM.h"
#include "unionFind.h"
#include <cstdio>
#include <mat.h>

void scanning(unordered_map<string, double> &param, vector<Region> &region, vector<vector<int>> &blockMap, list<Vec2i> &adjacent,const Mat &lab, const Mat &img, vector<Mat> &map, vector<Mat> &result)
{
    clock_t s, e;
    s = clock();

	const double &alpha = param["alpha"];	// weighting of color
	const double &beta = param["beta"];		// weighting of texture
	const double &gamma = param["gamma"];	// weighting of border

	const float phi = 0.7f;                 // compute JND

	const float t = 1.7f;                   // compute DN

	const int T = 6;                        //
	const int D = 8;                        // compute DT
	const int TextureWindow = 3;            //

	const int SpatialWindow = 2;            // compute SI

	const int &M = lab.rows;
	const int &N = lab.cols;
    int NumSeg = region.size();

	{

		Mat gray = img;
		cvtColor(img, gray, COLOR_BGR2GRAY);
		//compute WLD for the entire image
		int R1 = 1, R2 = 2, R3 = 3;
		vector<int> f1 = {
			1, 1, 1,
			1, -8, 1,
			1, 1, 1
		};
		vector<int> f2 = {
			1, 1, 1 ,1, 1,
			1 ,0, 0, 0, 1,
			1, 0, -16 ,0 ,1,
			1, 0, 0, 0, 1,
			1, 1, 1, 1, 1
		};
		vector<int> f3 = {
			1, 1, 1 ,1 ,1, 1, 1,
			1, 0, 0, 0, 0, 0, 1,
			1 ,0 ,0, 0, 0, 0, 1,
			1, 0, 0, -24, 0 ,0 ,1,
			1, 0, 0, 0, 0, 0, 1,
			1, 0, 0, 0, 0, 0, 1,
			1, 1, 1, 1, 1, 1, 1
		};

		vector<vector<int>> F1(M), F2(M), F3(M);
		vector<vector<int>> CascadeF1(M*N), CascadeF2(M*N), CascadeF3(M*N);
		vector<vector<float>> W1(NumSeg), W2(NumSeg), W3(NumSeg);
		int K = 2 * TextureWindow + 1;

		for (int i = 0; i < M; i++)
		{
			F1[i].resize(N);
			F2[i].resize(N);
			F3[i].resize(N);
		}
		WLD(gray, F1, M, N, R1, f1, T, D);
		WLD(gray, F2, M, N, R2, f2, T, D);
		WLD(gray, F3, M, N, R3, f3, T, D);

		gray.release();

		for (int i = 0; i < M*N; i++)
		{
			CascadeF1[i].resize(K*K);
			CascadeF2[i].resize(K*K);
			CascadeF3[i].resize(K*K);
		}
		TextureCascadeWithSlidingWindow(F1, M, N, TextureWindow, K, CascadeF1);
		TextureCascadeWithSlidingWindow(F2, M, N, TextureWindow, K, CascadeF2);
		TextureCascadeWithSlidingWindow(F3, M, N, TextureWindow, K, CascadeF3);

		for (int i = 0; i < NumSeg; i++)
		{
			W1[i].resize(T*D);
			W2[i].resize(T*D);
			W3[i].resize(T*D);
			for (int j = 0; j < T*D; j++)
			{
				W1[i][j] = 0;
				W2[i][j] = 0;
				W3[i][j] = 0;
			}
		}
		// NAM-CXR-light: use matching CascadeF per scale (Linux original bug used F1 thrice).
		AccumulateRangeofTexture(blockMap, CascadeF1, M, N, K, W1);
		AccumulateRangeofTexture(blockMap, CascadeF2, M, N, K, W2);
		AccumulateRangeofTexture(blockMap, CascadeF3, M, N, K, W3);

		TextureNormalized(W1, NumSeg, T, D);
		TextureNormalized(W2, NumSeg, T, D);
		TextureNormalized(W3, NumSeg, T, D);
		//compute WLD for each WLD
		for (int i = 0; i < NumSeg; i++)
		{
			Region &cur = region[i];
			for (int j = 0; j < T*D; j++)
			{
				cur.WLD[0][j] = W1[i][j];
				cur.WLD[1][j] = W2[i][j];
				cur.WLD[2][j] = W3[i][j];
			}
		}

		// N2: export area-weighted image-level WLD bins (3×48 = 144).
		if (!g_NAMFeatOutPath.empty())
		{
			vector<float> feat(3 * T * D, 0.f);
			float areaSum = 0.f;
			for (int i = 0; i < NumSeg; i++)
				areaSum += region[i].area;
			if (areaSum <= 0.f) areaSum = 1.f;
			for (int i = 0; i < NumSeg; i++)
			{
				float w = region[i].area / areaSum;
				for (int s = 0; s < 3; s++)
					for (int j = 0; j < T * D; j++)
						feat[s * T * D + j] += w * region[i].WLD[s][j];
			}
			FILE *fp = fopen(g_NAMFeatOutPath.c_str(), "w");
			if (fp)
			{
				for (size_t k = 0; k < feat.size(); k++)
					fprintf(fp, "%s%.8g", (k ? " " : ""), feat[k]);
				fprintf(fp, "\n");
				fclose(fp);
			}
		}
	}

	vector<int> parent(NumSeg);
	for (int i = 0; i < NumSeg; i++) parent[i] = i;
	list<float> Sum, Count, SI, DBC, DC, DAB, DW, DT, DN;
	float SumDC, SumDBC, SumDT;//unify the measurement range of DC DBC and DT
	float ThNc = (float)0.04*M*N;

	//initialize the features
	ComputeSumAndCount(lab, blockMap, M, N, NumSeg, Sum, Count, adjacent);
	ComputeSI(blockMap, M, N, NumSeg, SI, SpatialWindow, adjacent);
	ComputeDBC(Sum, Count, DBC);
	ComputeDC(region, adjacent, DC);
	ComputeDAB(region, adjacent, DAB);
	ComputeDW(region, adjacent, DW);
	ComputeDT(DAB, DW, DT);
	ComputeDN(region, adjacent, DN, t, ThNc);
	SumDC = SumDBC = SumDT = 0;
	for (auto &x : DC)
		SumDC += x;
	for (auto &x : DBC)
		SumDBC += x;
	for (auto &x : DT)
		SumDT += x;


	list<Vec2i>::iterator adjItr;
	list<float>::iterator sumItr, countItr, siItr, dbcItr, dcItr, dabItr, dwItr, dtItr, dnItr, dxyItr;

	//dist threshold
    float THRESHOLD;
    const double &Percent = param["percent"];

	int CountJ = 1;
	float JNDTHreshold = min((phi + 0.1f*CountJ)*2.3f, 2.3f);

	{
		list<float> Dist;
		ComputeDist(param,DC, DBC, DT, SI, DN, SumDC, SumDBC, SumDT, alpha, beta, gamma, Dist);
		ComputeThreshold(Dist, Percent, THRESHOLD);
    }

	while (NumSeg > param["change"])
	{
		adjItr = adjacent.begin();
		siItr = SI.begin();
		dbcItr = DBC.begin();
		dcItr = DC.begin();
		dtItr = DT.begin();
		dnItr = DN.begin();
		while (adjItr != adjacent.end())
		{
			//DBC<JNDTHRESHOLD
			if ((*dbcItr) < JNDTHreshold)break;

			//Dist<THRESHOLD
			float d;
			ComputeDist(param,dcItr, dbcItr, dtItr, siItr, dnItr, SumDC, SumDBC, SumDT, alpha, beta, gamma, d);
			if (d < THRESHOLD)break;

			adjItr++;
			siItr++;
			dbcItr++;
			dcItr++;
			dtItr++;
			dnItr++;
		}

		//recompute dist threshold
		if (adjItr == adjacent.end())
		{
			list<float> Dist;
			ComputeDist(param,DC, DBC, DT, SI, DN, SumDC, SumDBC, SumDT, alpha, beta, gamma, Dist);
			ComputeThreshold(Dist, Percent, THRESHOLD);

            CountJ += 1;
			continue;
		}
		//regions to be merged
		const Vec2i &merge = (*adjItr);
		int R1 = min(merge[0], merge[1]);
		int R2 = max(merge[0], merge[1]);

        //update regions
		NumSeg--;
		UpdateRegion(region, parent, R1, R2);

		//update adjacency list and features
		UpdateProperty(adjacent, DC, DBC, DT, SI, DN, Sum, Count, DAB, DW, SumDC, SumDT, SumDBC, region, parent, R1, R2, t, ThNc);
	}
	while (NumSeg > 1)
	{

		float MIN = INT_MAX;
		list<float> Dist;
		ComputeDist(param,DC, DBC, DT, SI, DN, SumDC, SumDBC, SumDT, alpha, beta, gamma, Dist);
		adjItr = adjacent.begin();
		list<float>::iterator distItr = Dist.begin();
		list<Vec2i>::iterator pos;
		while (distItr != Dist.end())
		{
			if (MIN > (*distItr))
			{
				MIN = (*distItr);
				pos = adjItr;
			}
			distItr++;
			adjItr++;
		}

		//regions to be merged
		const Vec2i &merge = (*pos);
		int R1 = min(merge[0], merge[1]);
		int R2 = max(merge[0], merge[1]);

		//update region
		NumSeg--;
		UpdateRegion(region, parent, R1, R2);

		//update adjacency list and features
		UpdateProperty(adjacent, DC, DBC, DT, SI, DN, Sum, Count, DAB, DW, SumDC, SumDT, SumDBC, region, parent, R1, R2, t, ThNc);

		if (NumSeg <= param["saveHir"])
		{
			//get latest blockMap
			unordered_map<int, int> LabelForOutput;
			int index = 1;
			for (int i = 0; i < parent.size(); i++)
				if (parent[i] == i)
					LabelForOutput[i] = index++;

			//output label matrix
			Mat m = Mat::zeros(M, N, CV_32FC1);
			for (int i = 0; i < M; i++)
				for (int j = 0; j < N; j++)
					m.at<float>(i, j) = LabelForOutput[Find_Parent(parent, blockMap[i][j])];
			map.push_back(std::move(m));
			m.release();

			if (param["mode"] == 1) {
			    //output color image
                Mat r = Mat::zeros(M, N, CV_32FC3);
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++)
                        r.at<Vec3f>(i, j) = region[Find_Parent(parent, blockMap[i][j])].lab;
                cvtColor(r, r, COLOR_Lab2BGR);
                r.convertTo(r, CV_8UC3, 255.0, 0);
                result.push_back(std::move(r));
                r.release();
            }
		}
	}
	e = clock();
	param["scanning"] = (e - s) / (float)CLOCKS_PER_SEC;
}
