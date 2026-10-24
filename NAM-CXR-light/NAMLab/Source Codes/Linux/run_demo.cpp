/*************************************************
author: Bowen Yang
date:2021/01/10
describe: demo
**************************************************/
#include <sys/stat.h>
#include "NAM.h"
#include "convertToMat.h"

// ./run_demo imagePath mode saveHir
// ./run_demo ./1.jpg 1 60
int main(int argc, char * argv[])
{
	if (argc < 4)	return 0;
	string file = argv[1];
	string inDir = file.substr(0, file.rfind('/'));
	inDir = inDir + '/';
	string x = file.substr(file.rfind('/') + 1, file.size());

	unordered_map<string, double> param;

	/* Output */
	param["mode"] = atof(argv[2]);		// 0: Only label 1: Output images and label
	param["saveHir"] = atof(argv[3]);	// Number of saved images (in our paper it is 60)

	/* Rgb2lab */
	param["lab"] = 0;					// Light source 0:D50 1:D65

	/* Representation */
	param["homogeneous"] = 0;			// Dissimilarity between two pixels. 0:Euclidean Eq.(10) 1:Gouraud Eq. (11)
	param["k"] = 2.0;					// Threshold in Eq. (10) or Eq. (11)
	param["shape"] = 0;					// Shape of homogeneous blocks 0:Square 1:Diagonal 2:Vertical 3:Diagonal
	param["ratio"] = 1.0;				// Maximum ratio of height to width of homogeneous blocks

	/* Mergence */
	param["u"] = 1.9;					// Mean
	param["var"] = 3.0;					// Variance 

	/* Scanning */
	param["alpha"] = 1.0;				//
	param["beta"] = 1.97;				//
	param["gamma"] = 1.97;				//
	param["percent"] = 6.7;				// Scanning
	param["si"] = 67.0;					//
	param["size"] = 4700.0;				//
	param["change"] = 197;				//

	/* Display in terminal */
	param["seg"] = 0;					// Number of segments after removing remnant regions
	param["remnant"] = 0;
	param["scanning"] = 0;				// Cost time
	param["representation"] = 0;
	param["mergence"] = 0;

	vector<Mat> map;
	vector<Mat> result;

	clock_t s, e;
	s = clock();

	// NAM-CXR-light: ensure Output exists before NAM writes *_wld144.txt
	mkdir((inDir + "Output").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	mkdir("Output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

	NAM(param, inDir + x, map, result);

	if (param["mode"] ==  0)
	{
		convertToMat(map, inDir + "Output/" + x.substr(0, x.size() - 4) + ".mat");
		for (auto &x : map)
			x.release();
	}
	else if (param["mode"] == 1)
	{
		Mat img = imread(inDir + x, IMREAD_UNCHANGED);
		if (!img.data)
			return -1;
		if (img.channels() == 1)
			cvtColor(img, img, COLOR_GRAY2BGR);
		else if (img.channels() == 4)
			cvtColor(img, img, COLOR_BGRA2BGR);
		else if (img.channels() != 3)
			return -1;
		const int &M = img.rows;
		const int &N = img.cols;

		for (int i = 0; i < result.size(); i++)
		{
			Mat mark = Mat::zeros(M, N, CV_8UC1);
			Mat mask = img.clone();
			Mat &mean = result[result.size() - 1 - i];

			imwrite(inDir + "Output/" + x.substr(0, x.size() - 4) + string("_Hier_") + to_string(i + 1) + string("_MeanNoBoundaryLine") + ".png", mean);
			for (int m = 0; m < M; m++)
				for (int n = 0; n < N; n++)
					if (!mark.at<uchar>(m, n)&&m >= 1 && m < M - 1 && map[map.size() - 1 - i].at<float>(m - 1, n) != map[map.size() - 1 - i].at<float>(m + 1, n) && !mark.at<uchar>(m - 1, n) && !mark.at<uchar>(m + 1, n)
						|| n >= 1 && n < N - 1 && map[map.size() - 1 - i].at<float>(m, n - 1) != map[map.size() - 1 - i].at<float>(m, n + 1) && !mark.at<uchar>(m, n - 1) && !mark.at<uchar>(m, n + 1))
					{
						mark.at<uchar>(m, n) = 1;
						mask.at<Vec3b>(m, n) = Vec3b(0, 0, 255);
						mean.at<Vec3b>(m, n) = Vec3b(0, 0, 255);
					}

			imwrite(inDir + "Output/" + x.substr(0, x.size() - 4) + string("_Hier_") + to_string(i + 1) + string("_Mask") + ".png", mask);
			imwrite(inDir + "Output/" + x.substr(0, x.size() - 4) + string("_Hier_") + to_string(i + 1) + string("_Mean") + ".png", mean);

			mean.release();
			mask.release();
			mark.release();
		}

		convertToMat(map, inDir + "Output/" + x.substr(0, x.size() - 4) + ".mat");
		for (auto &x : map)
			x.release();
	}

	e = clock();

	cout << "segment: " << param["seg"] << endl;
	cout << "representation time: " << param["representation"] << " s" << endl;
	cout << "mergence time: " << param["mergence"] << " s" << endl;
	cout << "remnantRemove time: " << param["remnant"] << " s" << endl;
	cout << "scanning time: " << param["scanning"] << " s" << endl;
	cout << "total time: " << (e - s) / (float)CLOCKS_PER_SEC << " s" << endl;
	return 0;
}
