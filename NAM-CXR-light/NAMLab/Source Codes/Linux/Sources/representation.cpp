/*************************************************
author: Bowen Yang
date:2019/10/22
describe: represent the image using a rectangular sub-pattern.
**************************************************/

#include "NAM.h"

static inline bool pixelCmp(float L1, float L2, float A1, float A2, float B1, float B2, float threshold);
static inline bool ratioCmp(double x1, double y1, double x2, double y2, double threshold);
static int Euler(const Mat &lab, const Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt, const float &K, const int &method = 0);
static int Gouraud(const Mat &lab, const Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt, const float &K, const int &method = 0);
static void mark(Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt);
static Vec2i getMaxBlock(const Mat &lab, const Mat &Mark, const int &M, const int &N, const Vec2i &start, const float &K, const double &ratio, const int &shape_method, const int &homogeneous_method);
static Vec3d blkColor(const Mat &lab, const Coordinate &blk);
static Vec3d blkVariance(const Mat &lab, const Coordinate &blk);
static void ColorVarianceInit(const Mat &lab, const vector<Coordinate> &P, vector<ColorVariance> &R);

void representation(unordered_map<string, double> &param,const Mat &lab, vector<Coordinate> &P, vector<ColorVariance> &R,vector<vector<int>> &blockMap)
{
    clock_t s, e;
    s = clock();

    const int &M = lab.rows;
    const int &N = lab.cols;
    Vec2i start;
    Mat Mark = Mat::zeros(M, N, CV_8UC1);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
        {
            if (Mark.at<uchar>(i, j))continue;
            start = Vec2i(i, j);
            Coordinate coordinate;
            coordinate.EP = getMaxBlock(lab, Mark, M, N, start, param["k"], param["ratio"], param["shape"], param["homogeneous"]);
            coordinate.SP = start;
            coordinate.area  = (coordinate.EP[0] - coordinate.SP[0] + 1)*(coordinate.EP[1] - coordinate.SP[1] + 1);
            P.push_back(coordinate);
            for (int m = coordinate.SP[0]; m <= coordinate.EP[0]; m++)
                for (int n = coordinate.SP[1]; n <= coordinate.EP[1]; n++)
                    blockMap[m][n] = P.size() - 1;
            mark(Mark, start, coordinate.EP);
        }
    ColorVarianceInit(lab,P,R);
    Mark.release();

    e = clock();
    param["representation"] = (e - s) / (float)CLOCKS_PER_SEC;
}

static inline bool pixelCmp(float L1, float L2, float A1, float A2, float B1, float B2, float threshold)
{
    // NAM-CXR-light: intensity-only homogeneity (a*/b* unused on CXR).
    (void)A1; (void)A2; (void)B1; (void)B2;
    return fabs(L1 - L2) >= threshold;
}
static inline bool ratioCmp(double x1, double y1, double x2, double y2, double threshold)
{
    double MAX, MIN;
    MAX = max(x2 - x1 + 1, y2 - y1 + 1);
    MIN = min(x2 - x1 + 1, y2 - y1 + 1);
    return (MAX / MIN) > threshold;
}

static int Euler(const Mat &lab, const Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt,const float &K, const int &method)
{
    float L1, A1, B1, L2, A2, B2, L3, A3, B3;
    int i, j;
    switch (method)
    {
        /* diagonal growth */
        case 0:
        {
            if (Mark.at<uchar>(right_bottom_pt)) return 0;

            j = right_bottom_pt[1];
            for (i = left_top_pt[0]; i < right_bottom_pt[0]; i++)
                if (Mark.at<uchar>(i, j)) return 0;

            i = right_bottom_pt[0];
            for (j = left_top_pt[1]; j < right_bottom_pt[1]; j++)
                if (Mark.at<uchar>(i, j)) return 0;

            j = right_bottom_pt[1];
            for (i = left_top_pt[0]; i < right_bottom_pt[0]; i++)
            {
                L1 = lab.at<Vec3f>(i, j)[0];
                A1 = lab.at<Vec3f>(i, j)[1];
                B1 = lab.at<Vec3f>(i, j)[2];

                L2 = lab.at<Vec3f>(i, j - 1)[0];
                A2 = lab.at<Vec3f>(i, j - 1)[1];
                B2 = lab.at<Vec3f>(i, j - 1)[2];

                L3 = lab.at<Vec3f>(i + 1, j)[0];
                A3 = lab.at<Vec3f>(i + 1, j)[1];
                B3 = lab.at<Vec3f>(i + 1, j)[2];

                if (pixelCmp(L1, L2, A1, A2, B1, B2, K) || pixelCmp(L1, L3, A1, A3, B1, B3, K)) return 0;
            }

            i = right_bottom_pt[0];
            for (j = left_top_pt[1]; j < right_bottom_pt[1]; j++)
            {
                L1 = lab.at<Vec3f>(i, j)[0];
                A1 = lab.at<Vec3f>(i, j)[1];
                B1 = lab.at<Vec3f>(i, j)[2];

                L2 = lab.at<Vec3f>(i - 1, j)[0];
                A2 = lab.at<Vec3f>(i - 1, j)[1];
                B2 = lab.at<Vec3f>(i - 1, j)[2];

                L3 = lab.at<Vec3f>(i, j + 1)[0];
                A3 = lab.at<Vec3f>(i, j + 1)[1];
                B3 = lab.at<Vec3f>(i, j + 1)[2];

                if (pixelCmp(L1, L2, A1, A2, B1, B2, K) || pixelCmp(L1, L3, A1, A3, B1, B3, K)) return 0;
            }
            break;
        }
            /* horizontal growth */
        case 1:
        {
            j = right_bottom_pt[1];
            for (i = left_top_pt[0]; i <= right_bottom_pt[0]; i++)
                if (Mark.at<uchar>(i, j)) return 0;

            j = right_bottom_pt[1];
            for (i = left_top_pt[0]; i < right_bottom_pt[0]; i++)
            {
                L1 = lab.at<Vec3f>(i, j)[0];
                A1 = lab.at<Vec3f>(i, j)[1];
                B1 = lab.at<Vec3f>(i, j)[2];

                L2 = lab.at<Vec3f>(i, j - 1)[0];
                A2 = lab.at<Vec3f>(i, j - 1)[1];
                B2 = lab.at<Vec3f>(i, j - 1)[2];

                L3 = lab.at<Vec3f>(i + 1, j)[0];
                A3 = lab.at<Vec3f>(i + 1, j)[1];
                B3 = lab.at<Vec3f>(i + 1, j)[2];

                if (pixelCmp(L1, L2, A1, A2, B1, B2, K) || pixelCmp(L1, L3, A1, A3, B1, B3, K)) return 0;
            }
            L1 = lab.at<Vec3f>(i, j)[0];
            A1 = lab.at<Vec3f>(i, j)[1];
            B1 = lab.at<Vec3f>(i, j)[2];
            L2 = lab.at<Vec3f>(i, j - 1)[0];
            A2 = lab.at<Vec3f>(i, j - 1)[1];
            B2 = lab.at<Vec3f>(i, j - 1)[2];
            if (pixelCmp(L1, L2, A1, A2, B1, B2, K)) return 0;
            break;
        }
            /* vertical growth*/
        case 2:
        {
            i = right_bottom_pt[0];
            for (j = left_top_pt[1]; j <= right_bottom_pt[1]; j++)
                if (Mark.at<uchar>(i, j)) return 0;

            i = right_bottom_pt[0];
            for (j = left_top_pt[1]; j < right_bottom_pt[1]; j++)
            {
                L1 = lab.at<Vec3f>(i, j)[0];
                A1 = lab.at<Vec3f>(i, j)[1];
                B1 = lab.at<Vec3f>(i, j)[2];

                L2 = lab.at<Vec3f>(i - 1, j)[0];
                A2 = lab.at<Vec3f>(i - 1, j)[1];
                B2 = lab.at<Vec3f>(i - 1, j)[2];

                L3 = lab.at<Vec3f>(i, j + 1)[0];
                A3 = lab.at<Vec3f>(i, j + 1)[1];
                B3 = lab.at<Vec3f>(i, j + 1)[2];

                if (pixelCmp(L1, L2, A1, A2, B1, B2, K) || pixelCmp(L1, L3, A1, A3, B1, B3, K)) return 0;
            }
            L1 = lab.at<Vec3f>(i, j)[0];
            A1 = lab.at<Vec3f>(i, j)[1];
            B1 = lab.at<Vec3f>(i, j)[2];
            L2 = lab.at<Vec3f>(i - 1, j)[0];
            A2 = lab.at<Vec3f>(i - 1, j)[1];
            B2 = lab.at<Vec3f>(i - 1, j)[2];
            if (pixelCmp(L1, L2, A1, A2, B1, B2, K)) return 0;
            break;
        }
    }
    return 1;
}

static int Gouraud(const Mat &lab, const Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt,const float &K, const int &method)
{
    const int &x1 = left_top_pt[1];
    const int &y1 = left_top_pt[0];
    const int &x2 = right_bottom_pt[1];
    const int &y2 = right_bottom_pt[0];
    const Vec3f &g1 = lab.at<Vec3f>(Point(x1, y1));
    const Vec3f &g2 = lab.at<Vec3f>(Point(x2, y1));
    const Vec3f &g3 = lab.at<Vec3f>(Point(x1, y2));
    const Vec3f &g4 = lab.at<Vec3f>(Point(x2, y2));
    //standard rectangle
    if (y1 < y2 && x1 < x2)
    {
        int flag = 1;
        for (int i = y1; i <= y2; i++)
        {
            for (int j = x1; j <= x2; j++)
            {
                float i1 = (i - y1) / (float)(y2 - y1);
                float i2 = (j - x1) / (float)(x2 - x1);
                Vec3f g5 = g1 + (g2 - g1)*i2;
                Vec3f g6 = g3 + (g4 - g3)*i2;
                Vec3f gest = g5 + (g6 - g5)*i1;
                Vec3f result = (lab.at<Vec3f>(i, j) - gest);
                // NAM-CXR-light: Gouraud check on L* only.
                if(abs(result[0]) > K || Mark.at<uchar>(i, j) == 1)
                {
                    flag = 0;
                    break;
                }
            }
            if (flag == 0)break;
        }
        return flag;
    }
    //rectangle with width 1
    else if (x1 != x2 && y1 == y2)
    {
        int flag = 1;
        for (int j = x1; j <= x2; j++)
        {
            float i2 = (j - x1) / (float)(x2 - x1);
            Vec3f gest = g1 + (g4 - g1)*i2;
            Vec3f result = (lab.at<Vec3f>(y1, j) - gest);
            // NAM-CXR-light: Gouraud check on L* only.
            if(abs(result[0]) > K || Mark.at<uchar>(y1, j) == 1)
            {
                flag = 0;
                break;
            }
        }
        return flag;
    }
    //rectangle with height 1
    else if (x1 == x2 && y1 != y2)
    {
        int flag = 1;
        for (int i = y1; i <= y2; i++)
        {
            float i1 = (i - y1) / (float)(y2 - y1);
            Vec3f gest = g1 + (g4 - g1)*i1;
            Vec3f result = (lab.at<Vec3f>(i, x1) - gest);

            // NAM-CXR-light: Gouraud check on L* only.
            if(abs(result[0]) > K || Mark.at<uchar>(i, x1) == 1)
            {
                flag = 0;
                break;
            }
        }
        return flag;
    }
    //isolated point
    else if (x1 == x2 && y1 == y2)
    {
        if (Mark.at<uchar>(left_top_pt))return 0;
        return 1;
    }
}

static void mark(Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt)
{
    for (int i = left_top_pt[0]; i <= right_bottom_pt[0]; i++)
        for (int j = left_top_pt[1]; j <= right_bottom_pt[1]; j++)
            Mark.at<uchar>(i, j) = 1;
}

static Vec2i getMaxBlock(const Mat &lab, const Mat &Mark, const int &M, const int &N, const Vec2i &start, const float &K, const double &ratio, const int &shape_method, const int &homogeneous_method)
{
    int(*isSameBlock)(const Mat &lab, const Mat &Mark, const Vec2i &left_top_pt, const Vec2i &right_bottom_pt,const float &K, const int &method);
    if (homogeneous_method == 0)
        isSameBlock = Euler;
    else
        isSameBlock = Gouraud;
    Vec2i end = start;
    switch (shape_method)
    {
        /* square subpattern */
        case 0:
        {
            end[0]++;
            end[1]++;
            while (end[0] < M && end[1] < N)
            {
                if (!isSameBlock(lab, Mark, start, end, K, 0)) break;
                end[0]++;
                end[1]++;
            }
            end[0]--;
            end[1]--;
            break;
        }
        /* diagonal priority */
        case 1:
        {
            end[0]++;
            end[1]++;
            while (end[0] < M && end[1] < N)
            {
                if (!isSameBlock(lab, Mark, start, end, K, 0)) break;
                end[0]++;
                end[1]++;
            }
            end[0]--;
            end[1]--;

            Vec2i end1 = end;
            end1[1]++;
            while (end1[1] < N) {
                if (!isSameBlock(lab, Mark, start, end1, K, 1)) break;
                if (ratioCmp(start[0], start[1], end1[0], end1[1], ratio)) break;
                end1[1]++;
            }
            end1[1]--;

            Vec2i end2 = end;
            end2[0]++;
            while (end2[0] < M) {
                if (!isSameBlock(lab, Mark, start, end2, K, 2)) break;
                if (ratioCmp(start[0], start[1], end2[0], end2[1], ratio)) break;
                end2[0]++;
            }
            end2[0]--;

            if ((end2[0] - start[0]) < (end1[1] - start[1])) end = end1;
            else end = end2;
            break;
        }
        /* horizontal priority */
        case 2:
        {
            end[1]++;
            while (end[1] < N) {
                if (!isSameBlock(lab, Mark, start, end, K, 1)) break;
                if (ratioCmp(start[0], start[1], end[0], end[1], ratio)) break;
                end[1]++;
            }
            end[1]--;

            end[0]++;
            while (end[0] < M) {
                if (!isSameBlock(lab, Mark, start, end, K, 2)) break;
                if (ratioCmp(start[0], start[1], end[0], end[1], ratio)) break;
                end[0]++;
            }
            end[0]--;
            break;
        }
        /* vertical priority */
        case 3:
        {

            end[0]++;
            while (end[0] < M) {
                if (!isSameBlock(lab, Mark, start, end, K, 2)) break;
                if (ratioCmp(start[0], start[1], end[0], end[1], ratio)) break;
                end[0]++;
            }
            end[0]--;

            end[1]++;
            while (end[1] < N) {
                if (!isSameBlock(lab, Mark, start, end, K, 1)) break;
                if (ratioCmp(start[0], start[1], end[0], end[1], ratio)) break;
                end[1]++;
            }
            end[1]--;
            break;
        }
    }
    return end;
}

static Vec3d blkColor(const Mat &lab, const Coordinate &coordinate)
{
    Vec3d color = lab.at<Vec3f>(coordinate.SP);
    float n = 1;
    if (coordinate.SP[1] < coordinate.EP[1])
    {
        color += lab.at<Vec3f>(coordinate.SP[0], coordinate.EP[1]);
        n++;
        if (coordinate.SP[0] < coordinate.EP[0])
        {
            color += lab.at<Vec3f>(coordinate.EP);
            n++;
            color += lab.at<Vec3f>(coordinate.EP[0], coordinate.SP[1]);
            n++;
        }
    }
    else if (coordinate.SP[0] < coordinate.EP[0])
    {
        color += lab.at<Vec3f>(coordinate.EP[0], coordinate.SP[1]);
        n++;
    }
    color /= n;
    return color;
}

static Vec3d blkVariance(const Mat &lab,const Coordinate &coordinate)
{
    float h = coordinate.EP[0] - coordinate.SP[0] + 1;
    float w = coordinate.EP[1] - coordinate.SP[1] + 1;

    float C1 = (2 * w - 1) / (6 * (w - 1));
    float C2 = (2 * h - 1) / (6 * (h - 1));

    const Vec3f &g1 = lab.at<Vec3f>(coordinate.SP);
    const Vec3f &g2 = lab.at<Vec3f>(coordinate.SP[0], coordinate.EP[1]);
    const Vec3f &g3 = lab.at<Vec3f>(coordinate.EP[0], coordinate.SP[1]);
    const Vec3f &g4 = lab.at<Vec3f>(coordinate.EP);

    Vec3d var;

    if (coordinate.SP[0] < coordinate.EP[0] && coordinate.SP[1] < coordinate.EP[1])
    {
        for (int i = 0; i < 3; i++)
            var[i] = (0.5f - C1 - C2)*(g1[i] * g4[i] + g2[i] * g3[i])
                     + C1 * (g2[i] * g4[i] + g1[i] * g3[i])
                     + C2 * (g3[i] * g4[i] + g1[i] * g2[i])
                     + C1 * C2*pow((g4[i] - g3[i] - g2[i] + g1[i]), 2)
                     - pow((g1[i] + g2[i] + g3[i] + g4[i]) / (float)4, 2);
    }
    else if (coordinate.SP[0] == coordinate.EP[0] && coordinate.SP[1] != coordinate.EP[1])
    {
        for (int i = 0; i < 3; i++)
            var[i] = (C1 - 0.25f)*pow(g4[i] - g1[i], 2);
    }
    else if (coordinate.SP[0] != coordinate.EP[0] && coordinate.SP[1] == coordinate.EP[1])
    {
        for (int i = 0; i < 3; i++)
            var[i] = (C2 - 0.25f)*pow(g4[i] - g1[i], 2);
    }
    else
    {
        for (int i = 0; i < 3; i++)
            var[i] = 0;
    }
    return var;
}

static void ColorVarianceInit(const Mat &lab, const vector<Coordinate> &P, vector<ColorVariance> &R)
{
    const size_t &NumSeg = P.size();
    R.resize(NumSeg);
    for (int i = 0; i < NumSeg; i++)
    {
        ColorVariance &r = R[i];
        const Coordinate &blk = P[i];
        r.color = blkColor(lab, blk);
        r.variance = blkVariance(lab, blk);
    }
}
