/*************************************************
author: Bowen Yang
date:2019/05/07
describe: Convert the segmentation result to a mat file for evaluation.
**************************************************/
#pragma once
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <mat.h>

static void convertToMat(const std::vector<cv::Mat> &map, const std::string &filepath)
{
	// Guard: empty hierarchy (failed NAM / non-image) must not touch map[0].
	if (map.empty())
		return;

	MATFile *pMF = nullptr;
	mxArray *pMxArray = nullptr;

	pMF = matOpen(filepath.c_str(), "w");
	if (pMF == nullptr)return;
	size_t *dim = new size_t[2];
	dim[0] = 1;
	dim[1] = map.size();
	pMxArray = mxCreateCellArray(2, dim);
	delete[] dim;

	int M = map[0].rows;
	int N = map[0].cols;

	int count = 0;
	for (int k = map.size() - 1, count = 0; k >= 0; k--, count++)
	{
		double *temp = new double[M*N];
		mxArray *array = mxCreateDoubleMatrix(M, N, mxREAL);
		for (int j = 0; j < M; j++)
			for (int i = 0; i < N; i++)
				temp[i*M + j] = map[k].at<float>(j, i);
		memcpy((void *)(mxGetPr(array)), (void *)temp, sizeof(double) * M*N);
		mxSetCell(pMxArray, count, array);
		delete[] temp;
	}
	matPutVariable(pMF, "segs", pMxArray);
	mxDestroyArray(pMxArray);
	matClose(pMF);
}
